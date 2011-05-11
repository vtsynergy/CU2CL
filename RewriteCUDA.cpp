#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

#include "clang/Lex/PPCallbacks.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "clang/Lex/Preprocessor.h"

#include "clang/Rewrite/Rewriter.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>

using namespace clang;
using namespace llvm::sys::path;

namespace {

struct cmpDG {
    bool operator()(DeclGroupRef a, DeclGroupRef b) {
        SourceLocation aLoc = (a.isSingleDecl() ? a.getSingleDecl() : a.getDeclGroup()[0])->getLocStart();
        SourceLocation bLoc = (b.isSingleDecl() ? b.getSingleDecl() : b.getDeclGroup()[0])->getLocStart();
        return aLoc.getRawEncoding() < bLoc.getRawEncoding();
    }
};

class RewriteCUDA;

class RewriteIncludesCallback : public PPCallbacks {
private:
    RewriteCUDA *RCUDA;

public:
    RewriteIncludesCallback(RewriteCUDA *);

    virtual void InclusionDirective(SourceLocation, const Token &,
                                    llvm::StringRef, bool,
                                    const FileEntry *, SourceLocation/*,
                                    const llvm::SmallVectorImpl<char> &*/);

};

/**
 * An AST consumer made to rewrite CUDA to OpenCL.
 **/
class RewriteCUDA : public ASTConsumer, public PPCallbacks {
private:
    typedef std::map<FileID, llvm::raw_ostream *> IDOutFileMap;
    typedef std::map<llvm::StringRef, std::list<llvm::StringRef> > StringRefListMap;

    CompilerInstance *CI;
    SourceManager *SM;
    Preprocessor *PP;

    Rewriter HostRewrite;
    Rewriter KernelRewrite;

    //Rewritten files
    FileID MainFileID;
    llvm::raw_ostream *MainOutFile;
    llvm::raw_ostream *MainKernelOutFile;
    IDOutFileMap OutFiles;
    IDOutFileMap KernelOutFiles;
    //TODO lump IDs and both outfiles together

    StringRefListMap Kernels;

    std::set<DeclGroupRef, cmpDG> GlobalVarDeclGroups;
    std::set<DeclGroupRef, cmpDG> CurVarDeclGroups;
    std::set<DeclGroupRef, cmpDG> DeviceMemDGs;
    std::set<VarDecl *> DeviceMemVars;

    FunctionDecl *MainDecl;

    //Preamble string to insert at top of main host file
    std::string HostPreamble;
    std::string HostIncludes;
    std::string HostDecls;
    std::string HostGlobalVars;
    std::string HostKernels;
    std::string HostFunctions;

    //Preamble string to insert at top of main kernel file
    std::string DevPreamble;
    std::string DevFunctions;

    std::string CLInit;
    std::string CLClean;

    //Flags used by the rewriter
    bool IncludingStringH;
    bool UsesCUDADeviceProp;
    bool UsesCUDAMemset;

    void TraverseStmt(Stmt *e, unsigned int indent) {
        for (unsigned int i = 0; i < indent; i++)
            llvm::errs() << "  ";
        llvm::errs() << e->getStmtClassName() << "\n";
        indent++;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI)
            if (*CI)
                TraverseStmt(*CI, indent);
    }

    template <class T>
    T *FindStmt(Stmt *e) {
        if (T *t = dyn_cast<T>(e))
            return t;
        T *ret = NULL;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI) {
            ret = FindStmt<T>(*CI);
            if (ret)
                return ret;
        }
        return NULL;
    }

    void RewriteHostFunction(FunctionDecl *hostFunc) {
        //llvm::errs() << "Rewriting host function: " << hostFunc->getName() << "\n";
        //Remove from KernelRewrite
        RemoveFunction(hostFunc, KernelRewrite);
        //Rewrite __host__ attribute
        if (hostFunc->hasAttr<CUDAHostAttr>()) {
            SourceLocation loc = FindAttr(hostFunc->getLocStart(), "__host__");
            RewriteAttr("__host__", loc, HostRewrite);
        }
        if (Stmt *body = hostFunc->getBody()) {
            //TraverseStmt(body, 0);
            RewriteHostStmt(body);
        }
        CurVarDeclGroups.clear();
    }

    void RewriteHostStmt(Stmt *s) {
        //Traverse children and recurse
        for (Stmt::child_iterator CI = s->child_begin(), CE = s->child_end();
             CI != CE; ++CI) {
            if (*CI)
                RewriteHostStmt(*CI);
        }
        //Visit this node
        if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(s)) {
            RewriteCUDAKernelCall(kce);
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(s)) {
            if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0)
                RewriteCUDACall(ce);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
            DeclGroupRef DG = ds->getDeclGroup();
            Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            //Store VarDecl DeclGroupRefs
            if (firstDecl->getKind() == Decl::Var) {
                CurVarDeclGroups.insert(DG);
            }
            for(DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    RewriteVarDecl(vd);
                }
                //TODO other non-top level declarations??
            }
        }
        else if (MemberExpr *me = dyn_cast<MemberExpr>(s)) {
            //Check base Expr, if DeclRefExpr and a dim3, then rewrite
            if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(me->getBase())) {
                std::string type = dre->getDecl()->getType().getAsString();
                if (type == "dim3") {
                    std::string name = me->getMemberDecl()->getNameAsString();
                    if (name == "x") {
                        name = "[0]";
                    }
                    else if (name == "y") {
                        name = "[1]";
                    }
                    else if (name == "z") {
                        name = "[2]";
                    }
                    //TODO find location of arrow or dot and rewrite
                    SourceRange sr = me->getSourceRange();
                    HostRewrite.ReplaceText(sr.getBegin(), HostRewrite.getRangeSize(sr), PrintStmtToString(dre) + name);
                }
                else if (type == "cudaDeviceProp") {
                    //TODO check what the reference is
                    //TODO if unsupported, print a warning
                }
            }
        }
    }

    void RewriteKernelFunction(FunctionDecl *kernelFunc) {
        //llvm::errs() << "Rewriting CUDA kernel\n";
        bool hasHost = kernelFunc->hasAttr<CUDAHostAttr>();
        if (!hasHost) {
            RemoveFunction(kernelFunc, HostRewrite);
        }
        llvm::StringRef r = stem(SM->getFileEntryForID(SM->getFileID(kernelFunc->getLocation()))->getName());
        std::list<llvm::StringRef> &l = Kernels[r];
        l.push_back(kernelFunc->getName());
        HostKernels += "cl_kernel clKernel_" + kernelFunc->getName().str() + ";\n";
        //Rewrite kernel attributes
        if (kernelFunc->hasAttr<CUDAGlobalAttr>()) {
            SourceLocation loc = FindAttr(kernelFunc->getLocStart(), "__global__");
            RewriteAttr("__global__", loc, KernelRewrite);
            if (hasHost)
                RewriteAttr("__global__", loc, HostRewrite);
        }
        if (kernelFunc->hasAttr<CUDADeviceAttr>()) {
            SourceLocation loc = FindAttr(kernelFunc->getLocStart(), "__device__");
            RewriteAttr("__device__", loc, KernelRewrite);
            if (hasHost)
                RewriteAttr("__device__", loc, HostRewrite);
        }
        if (kernelFunc->hasAttr<CUDAHostAttr>()) {
            SourceLocation loc = FindAttr(kernelFunc->getLocStart(), "__host__");
            RewriteAttr("__host__", loc, KernelRewrite);
            if (hasHost)
                RewriteAttr("__host__", loc, HostRewrite);
        }
        //Rewrite arguments
        for (FunctionDecl::param_iterator PI = kernelFunc->param_begin(),
                                          PE = kernelFunc->param_end();
                                          PI != PE; ++PI) {
            RewriteKernelParam(*PI);
        }
        //Rewirte kernel body
        if (kernelFunc->hasBody()) {
            //TraverseStmt(kernelFunc->getBody(), 0);
            RewriteKernelStmt(kernelFunc->getBody());
        }
    }

    void RewriteKernelParam(ParmVarDecl *parmDecl) {
        if (parmDecl->hasAttr<CUDADeviceAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__device__");
            RewriteAttr("__device__", loc, KernelRewrite);
        }
        else if (parmDecl->hasAttr<CUDAConstantAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__constant__");
            RewriteAttr("__constant__", loc, KernelRewrite);
        }
        else if (parmDecl->hasAttr<CUDASharedAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__shared__");
            RewriteAttr("__shared__", loc, KernelRewrite);
        }
        else if (parmDecl->getOriginalType().getTypePtr()->isPointerType()) {
            KernelRewrite.InsertTextBefore(
                    parmDecl->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                    "__global ");
        }
    }

    void RewriteKernelStmt(Stmt *ks) {
        //Traverse children and recurse
        for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end();
             CI != CE; ++CI) {
            if (*CI)
                RewriteKernelStmt(*CI);
        }
        //Visit this node
        std::string str;
        if (MemberExpr *me = dyn_cast<MemberExpr>(ks)) {
            //TODO check base expr, if declrefexpr and a dim3, then rewrite
            str = PrintStmtToString(me);
            if (str == "threadIdx.x")
                ReplaceStmtWithText(ks, "get_local_id(0)", KernelRewrite);
            else if (str == "threadIdx.y")
                ReplaceStmtWithText(ks, "get_local_id(1)", KernelRewrite);
            else if (str == "threadIdx.z")
                ReplaceStmtWithText(ks, "get_local_id(2)", KernelRewrite);
            else if (str == "blockIdx.x")
                ReplaceStmtWithText(ks, "get_group_id(0)", KernelRewrite);
            else if (str == "blockIdx.y")
                ReplaceStmtWithText(ks, "get_group_id(1)", KernelRewrite);
            else if (str == "blockIdx.z")
                ReplaceStmtWithText(ks, "get_group_id(2)", KernelRewrite);
            else if (str == "blockDim.x")
                ReplaceStmtWithText(ks, "get_local_size(0)", KernelRewrite);
            else if (str == "blockDim.y")
                ReplaceStmtWithText(ks, "get_local_size(1)", KernelRewrite);
            else if (str == "blockDim.z")
                ReplaceStmtWithText(ks, "get_local_size(2)", KernelRewrite);
            else if (str == "gridDim.x")
                ReplaceStmtWithText(ks, "get_num_groups(0)", KernelRewrite);
            else if (str == "gridDim.y")
                ReplaceStmtWithText(ks, "get_num_groups(1)", KernelRewrite);
            else if (str == "gridDim.z")
                ReplaceStmtWithText(ks, "get_num_groups(2)", KernelRewrite);
        }
        /*else if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(ks)) {
        }*/
        else if (CallExpr *ce = dyn_cast<CallExpr>(ks)) {
            str = PrintStmtToString(ce);
            if (str == "__syncthreads()")
                ReplaceStmtWithText(ks, "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", KernelRewrite);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(ks)) {
            DeclGroupRef DG = ds->getDeclGroup();
            Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            //Store VarDecl DeclGroupRefs
            if (firstDecl->getKind() == Decl::Var) {
                CurVarDeclGroups.insert(DG);
            }
            for(DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    if (vd->hasAttr<CUDADeviceAttr>()) {
                        SourceLocation loc = FindAttr(vd->getLocStart(), "__device__");
                        RewriteAttr("__device__", loc, KernelRewrite);
                    }
                    else if (vd->hasAttr<CUDAConstantAttr>()) {
                        SourceLocation loc = FindAttr(vd->getLocStart(), "__constant__");
                        RewriteAttr("__constant__", loc, KernelRewrite);
                    }
                    else if (vd->hasAttr<CUDASharedAttr>()) {
                        SourceLocation loc = FindAttr(vd->getLocStart(), "__shared__");
                        RewriteAttr("__shared__", loc, KernelRewrite);
                    }
                }
                //TODO other non-top level declarations??
            }
        }
    }

    void RewriteCUDACall(CallExpr *cudaCall) {
        //llvm::errs() << "Rewriting CUDA API call\n";
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();
        if (funcName == "cudaThreadExit") {
            //Replace with clReleaseContext()
            ReplaceStmtWithText(cudaCall, "clReleaseContext(clContext)", HostRewrite);
        }
        else if (funcName == "cudaThreadSynchronize") {
            //Replace with clFinish
            ReplaceStmtWithText(cudaCall, "clFinish(clCommandQueue)", HostRewrite);
        }
        else if (funcName == "cudaGetDevice") {
            //Replace by assigning current value of clDevice to arg
            Expr *device = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(device);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            ReplaceStmtWithText(cudaCall, var->getNameAsString() + " = clDevice", HostRewrite);
        }
        else if (funcName == "cudaSetDevice") {
            //Replace with clCreateContext and clCreateCommandQueue
            Expr *device = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(device);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            std::string sub = "clContext = clCreateContext(NULL, 1,&" + var->getNameAsString() + ", NULL, NULL, NULL);\n";
            sub += "clCommandQueue = clCreateCommandQueue(clContext, " + var->getNameAsString() +", 0, NULL);\n";
            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }
        else if (funcName == "cudaMalloc") {
            //TODO check if the return value is being used somehow?
            Expr *varExpr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(varExpr);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            llvm::StringRef varName = var->getName();
            std::string str;
            str = PrintStmtToString(size);

            //Replace with clCreateBuffer
            std::string sub = varName.str() + " = clCreateBuffer(clContext, CL_MEM_READ_WRITE, " + str + ", NULL, NULL)";
            ReplaceStmtWithText(cudaCall, sub, HostRewrite);

            DeclGroupRef varDG(var);
            if (CurVarDeclGroups.find(varDG) != CurVarDeclGroups.end()) {
                DeviceMemDGs.insert(*CurVarDeclGroups.find(varDG));
            }
            else if (GlobalVarDeclGroups.find(varDG) != GlobalVarDeclGroups.end()) {
                DeviceMemDGs.insert(*GlobalVarDeclGroups.find(varDG));
            }
            else {
                //TODO single decl, so rewrite now as before
                //Change variable's type to cl_mem
                TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
                HostRewrite.ReplaceText(tl.getBeginLoc(),
                                    HostRewrite.getRangeSize(tl.getSourceRange()),
                                    "cl_mem ");
            }

            //Add var to DeviceMemVars
            DeviceMemVars.insert(var);
        }
        else if (funcName == "cudaFree") {
            Expr *devPtr = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(devPtr);
            llvm::StringRef varName = dr->getDecl()->getName();

            //Replace with clReleaseMemObject
            ReplaceStmtWithText(cudaCall, "clReleaseMemObject(" + varName.str() + ")", HostRewrite);
        }
        else if (funcName == "cudaMemcpy") {
            std::string replace;

            Expr *dst = cudaCall->getArg(0);
            Expr *src = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            Expr *kind = cudaCall->getArg(3);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(kind);
            EnumConstantDecl *enumConst = dyn_cast<EnumConstantDecl>(dr->getDecl());
            std::string enumString = enumConst->getNameAsString();
            //llvm::errs() << enumString << "\n";
            if (enumString == "cudaMemcpyHostToHost") {
                //standard memcpy
                //make sure to include <string.h>
                if (!IncludingStringH) {
                    HostIncludes += "#include <string.h>\n";
                    IncludingStringH = true;
                }
                replace += "memcpy(";
                replace += PrintStmtToString(dst) + ", ";
                replace += PrintStmtToString(src) + ", ";
                replace += PrintStmtToString(count) + ")";
                ReplaceStmtWithText(cudaCall, replace, HostRewrite);
            }
            else if (enumString == "cudaMemcpyHostToDevice") {
                //clEnqueueWriteBuffer
                replace += "clEnqueueWriteBuffer(clCommandQueue, ";
                replace += PrintStmtToString(dst) + ", CL_TRUE, 0, ";
                replace += PrintStmtToString(count) + ", ";
                replace += PrintStmtToString(src) + ", 0, NULL, NULL)";
                ReplaceStmtWithText(cudaCall, replace, HostRewrite);
            }
            else if (enumString == "cudaMemcpyDeviceToHost") {
                //clEnqueueReadBuffer
                replace += "clEnqueueReadBuffer(clCommandQueue, ";
                replace += PrintStmtToString(src) + ", CL_TRUE, 0, ";
                replace += PrintStmtToString(count) + ", ";
                replace += PrintStmtToString(dst) + ", 0, NULL, NULL)";
                ReplaceStmtWithText(cudaCall, replace, HostRewrite);
            }
            else if (enumString == "cudaMemcpyDeviceToDevice") {
                //TODO clEnqueueReadBuffer; clEnqueueWriteBuffer
                ReplaceStmtWithText(cudaCall, "clEnqueueReadBuffer(clCommandQueue, src, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
                ReplaceStmtWithText(cudaCall, "clEnqueueWriteBuffer(clCommandQueue, dst, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
            }
            else {
                //TODO Use diagnostics to print pretty errors
                llvm::errs() << "Unsupported cudaMemcpy type: " << enumString << "\n";
            }
        }
        else if (funcName == "cudaMemset") {
            if (!UsesCUDAMemset) {
                UsesCUDAMemset = true;
                HostFunctions += "cl_int clMemset(cl_mem devPtr, int value, size_t count) {\n" \
                                "\tclSetKernelArg(clKernel_clMemset, 0, sizeof(cl_mem), &devPtr);\n" \
                                "\tclSetKernelArg(clKernel_clMemset, 1, sizeof(int), &value);\n" \
                                "\tclSetKernelArg(clKernel_clMemset, 2, sizeof(size_t), &count);\n" \
                                "\tglobalWorkSize[0] = count;\n" \
                                "\treturn clEnqueueNDRangeKernel(clCommandQueue, clKernel_clMemset, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);\n" \
                                "}\n\n";
                DevFunctions += "__kernel void clMemset(__global unsigned char *ptr, unsigned char value, size_t num) {\n" \
                            "\tsize_t id = get_global_id(0);\n" \
                            "\tif (get_global_id(0) < num) {\n" \
                            "\t\tptr[id] = value;\n" \
                            "\t}\n" \
                            "}\n\n";
                llvm::StringRef r = stem(SM->getFileEntryForID(MainFileID)->getName());
                std::list<llvm::StringRef> &l = Kernels[r];
                l.push_back("clMemset");
                HostKernels += "cl_kernel clKernel_clMemset;\n";
            }
            //TODO follow Swan's example of setting via a kernel

            HostRewrite.ReplaceText(cudaCall->getLocStart(), funcName.length(), "clMemset");
        }
        else {
            //TODO Use diagnostics to print pretty errors
            llvm::errs() << "Unsupported CUDA call: " << funcName << "\n";
        }
    }

    void RewriteCUDAKernelCall(CUDAKernelCallExpr *kernelCall) {
        //llvm::errs() << "Rewriting CUDA kernel call\n";
        FunctionDecl *callee = kernelCall->getDirectCallee();
        CallExpr *kernelConfig = kernelCall->getConfig();
        std::string kernelName = "clKernel_" + callee->getNameAsString();
        std::ostringstream args;
        unsigned int dims = 1;
        for (unsigned i = 0; i < kernelCall->getNumArgs(); i++) {
            Expr *arg = kernelCall->getArg(i);
            VarDecl *var = dyn_cast<VarDecl>(FindStmt<DeclRefExpr>(arg)->getDecl());
            args << "clSetKernelArg(" << kernelName << ", " << i << ", sizeof(";
            if (DeviceMemVars.find(var) != DeviceMemVars.end()) {
                //arg var is a cl_mem
                args << "cl_mem";
            }
            else {
                args << var->getType().getAsString();
            }
            args << "), &" << var->getNameAsString() << ");\n";
        }
        //Guaranteed to be dim3s, so pull out their x,y,z values
        Expr *grid = kernelConfig->getArg(0);
        Expr *block = kernelConfig->getArg(1);
        //TraverseStmt(grid, 0);
        //TraverseStmt(block, 0);
        CXXConstructExpr *construct = dyn_cast<CXXConstructExpr>(block);
        ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
            //variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                dims = 3;
                for (unsigned int i = 0; i < 3; i++)
                    args << "localWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "];\n";
            }
            else {
                //TODO ??
                args << "localWorkSize[0] = " << PrintStmtToString(dre) << ";\n";
            }
        }
        else {
            //constant passed to block
            Expr *arg = cast->getSubExprAsWritten();
            args << "localWorkSize[0] = " << PrintStmtToString(arg) << ";\n";
        }
        construct = dyn_cast<CXXConstructExpr>(grid);
        cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
            //variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                dims = 3;
                for (unsigned int i = 0; i < 3; i++)
                    args << "globalWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "]*localWorkSize[" << i << "];\n";
            }
            else {
                //TODO ??
                args << "globalWorkSize[0] = (" << PrintStmtToString(dre) << ")*localWorkSize[0];\n";
            }
        }
        else {
            //constant passed to grid
            Expr *arg = cast->getSubExprAsWritten();
            args << "globalWorkSize[0] = (" << PrintStmtToString(arg) << ")*localWorkSize[0];\n";
        }
        args << "clEnqueueNDRangeKernel(clCommandQueue, " << kernelName << ", " << dims << ", NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)";
        ReplaceStmtWithText(kernelCall, args.str(), HostRewrite);
    }

    void RewriteMain(FunctionDecl *mainDecl) {
        MainDecl = mainDecl;
    }

    void RewriteVarDecl(VarDecl *var) {
        std::string type = var->getType().getAsString();
        if (type == "dim3") {
            //Rewrite to size_t array
            //TODO rewrite statement as a whole with a new string
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            HostRewrite.ReplaceText(tl.getBeginLoc(),
                                HostRewrite.getRangeSize(tl.getSourceRange()),
                                "size_t");
            if (var->hasInit()) {
                //Rewrite initial value
                CXXConstructExpr *ce = (CXXConstructExpr *) var->getInit();
                std::string args = " = {";
                for (CXXConstructExpr::arg_iterator i = ce->arg_begin(), e = ce->arg_end();
                     i != e; ++i) {
                    Expr *arg = *i;
                    if (CXXDefaultArgExpr *defArg = dyn_cast<CXXDefaultArgExpr>(arg)) {
                        args += PrintStmtToString(defArg->getExpr());
                    }
                    else {
                        args += PrintStmtToString(arg);
                    }
                    if (i + 1 != e)
                        args += ", ";
                }
                args += "}";
                SourceRange parenRange = ce->getParenRange();
                if (parenRange.isValid()) {
                    HostRewrite.ReplaceText(parenRange.getBegin(),
                                        HostRewrite.getRangeSize(parenRange),
                                        args);
                    HostRewrite.InsertTextBefore(parenRange.getBegin(), "[3]");
                }
                else {
                    HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(var->getLocEnd()), "[3]");
                    HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(var->getLocEnd()),
                                            args);
                }
            }
        }
        else if (type == "struct cudaDeviceProp") {
            if (!UsesCUDADeviceProp) {
                UsesCUDADeviceProp = true;
                HostDecls +=
                    "struct clDeviceProp {\n" \
                    "\tchar name[256];\n" \
                    "\tcl_ulong totalGlobalMem;\n" \
                    "\tcl_ulong sharedMemPerBlock;\n" \
                    "\tcl_uint regsPerBlock;\n" \
                    "\tcl_uint warpSize;\n" \
                    "\tint warpSize; //Unsupported!\n" \
                    "\tsize_t memPitch; //Unsupported!\n" \
                    "\tsize_t maxThreadsPerBlock;\n" \
                    "\tsize_t maxThreadsDim[3];\n" \
                    "\tint maxGridSize[3]; //Unsupported!\n" \
                    "\tcl_uint clockRate;\n" \
                    "\tsize_t totalConstMem; //Unsupported!\n" \
                    "\tcl_uint major;\n" \
                    "\tcl_uint minor;\n" \
                    "\tsize_t textureAlignment; //Unsupported!\n" \
                    "\tcl_bool deviceOverlap;\n" \
                    "\tcl_uint multiProcessorCount;\n" \
                    "\tcl_bool kernelExecTimeoutEnabled;\n" \
                    "\tcl_bool integrated;\n" \
                    "\tint canMapHostMemory; //Unsupported!\n" \
                    "\tint computeMode; //Unsupported!\n" \
                    "\tint maxTexture1D; //Unsupported!\n" \
                    "\tint maxTexture2D[2]; //Unsupported!\n" \
                    "\tint maxTexture3D[3]; //Unsupported!\n" \
                    "\tint maxTexture2DArray[3]; //Unsupported!\n" \
                    "\tsize_t surfaceAlignment; //Unsupported!\n" \
                    "\tint concurrentKernels; //Unsupported!\n" \
                    "\tcl_bool ECCEnabled;\n" \
                    "\tint pciBusID; //Unsupported!\n" \
                    "\tint pciDeviceID; //Unsupported!\n" \
                    "\tint tccDriver; //Unsupported!\n" \
                    "\t//int __cudaReserved[21];\n" \
                    "};\n\n";
                HostFunctions +=
                    "cl_int clGetDeviceProperties(struct clDeviceProp *prop, cl_device_id device) {\n" \
                    "\tcl_int ret = CL_SUCCESS;" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(prop->name), &prop->name, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(prop->totalGlobalMem), &prop->totalGlobalMem, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(prop->sharedMemPerBlock), &prop->sharedMemPerBlock, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(prop->regsPerBlock), &prop->regsPerBlock, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_WARP_SIZE_NV, sizeof(prop->warpSize), &prop->warpSize, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(prop->maxThreadsPerBlock), &prop->maxThreadsPerBlock, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(prop->maxThreadsDim), &prop->maxThreadsDim, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(prop->clockRate), &prop->clockRate, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(prop->major), &prop->major, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(prop->minor), &prop->minor, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_GPU_OVERLAP_NV, sizeof(prop->deviceOverlap), &prop->deviceOverlap, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(prop->multiProcessorCount), &prop->multiProcessorCount, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(prop->kernelExecTimeoutEnabled), &prop->kernelExecTimeoutEnabled, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_INTEGRATED_MEMORY_NV, sizeof(prop->integrated), &prop->integrated, NULL);\n" \
                    "\tret |= clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(prop->ECCEnabled), &prop->ECCEnabled, NULL);\n" \
                    "\treturn ret;\n" \
                    "}\n\n";
            }
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            HostRewrite.ReplaceText(tl.getBeginLoc(),
                                HostRewrite.getRangeSize(tl.getSourceRange()),
                                "clDeviceProp");
        }
        //TODO check other CUDA-only types to rewrite
    }

    void RewriteAttr(std::string attr, SourceLocation loc, Rewriter &Rewrite) {
        std::string replace;
        if (attr == "__global__") {
            replace = "__kernel";
        }
        else if (attr == "__device__") {
            replace = "";
        }
        else if (attr == "__host__") {
            replace = "";
        }
        else if (attr == "__constant__") {
            replace = "__constant";
        }
        else if (attr == "__shared__") {
            replace = "__local";
        }
        Rewrite.ReplaceText(loc, attr.length(), replace);
    }

    SourceLocation FindAttr(SourceLocation loc, std::string attr) {
        //TODO inefficient... find optimizations
        const char *s = attr.c_str();
        size_t len = attr.length();
        FileID fileID = SM->getFileID(loc);
        SourceLocation locStart = SM->getLocForStartOfFile(fileID);
        llvm::StringRef fileBuf = SM->getBufferData(fileID);
        const char *fileBufStart = fileBuf.begin();
        for (const char *bufPtr = fileBufStart + SM->getFileOffset(loc);
             bufPtr >= fileBufStart; bufPtr--) {
            if (strncmp(bufPtr, s, len) == 0)
                return locStart.getFileLocWithOffset(bufPtr - fileBufStart);
        }
        return SourceLocation();
    }

    void RemoveFunction(FunctionDecl *func, Rewriter &Rewrite) {
        SourceLocation startLoc, endLoc;
        //Find startLoc
        if (func->hasAttrs()) {
            Attr *attr = (func->getAttrs())[0];
            std::string attrStr;
            switch (attr->getKind()) {
                case attr::CUDAGlobal:
                    attrStr = "__global__";
                    break;
                case attr::CUDADevice:
                    attrStr = "__device__";
                    break;
                case attr::CUDAHost:
                    attrStr = "__host__";
                    break;
                default:
                    break;
            }
            startLoc = FindAttr(func->getLocStart(), attrStr);
        }
        else {
            //TODO find first specifier location
            startLoc = func->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        }
        //Find endLoc
        if (func->hasBody()) {
            CompoundStmt *body = (CompoundStmt *) func->getBody();
            endLoc = body->getRBracLoc();
        }
        else {
            //Find location of semi-colon
            endLoc = func->getSourceRange().getEnd();
        }
        Rewrite.RemoveText(startLoc,
                           Rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    void RemoveVar(VarDecl *var, Rewriter &Rewrite) {
        SourceLocation startLoc, endLoc;
        //Find startLoc
        if (var->hasAttrs()) {
            Attr *attr = (var->getAttrs())[0];
            std::string attrStr;
            switch (attr->getKind()) {
                case attr::CUDAGlobal:
                    attrStr = "__constant__";
                    break;
                case attr::CUDADevice:
                    attrStr = "__device__";
                    break;
                case attr::CUDAHost:
                    attrStr = "__shared__";
                    break;
                default:
                    break;
            }
            startLoc = FindAttr(var->getLocStart(), attrStr);
        }
        else {
            //TODO find first specifier location
            startLoc = var->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        }
        //Find endLoc
        if (var->hasInit()) {
            Expr *init = var->getInit();
            endLoc = init->getLocEnd();;
        }
        else {
            //Find location of semi-colon
            endLoc = var->getSourceRange().getEnd();
        }
        Rewrite.RemoveText(startLoc,
                           Rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    std::string PrintStmtToString(Stmt *s) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        s->printPretty(S, 0, PrintingPolicy(HostRewrite.getLangOpts()));
        return S.str();
    }

    std::string PrintDeclToString(Decl *d) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        d->print(S);
        return S.str();
    }

    bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr, Rewriter &Rewrite) {
        return Rewrite.ReplaceText(OldStmt->getLocStart(),
                                   Rewrite.getRangeSize(OldStmt->getSourceRange()),
                                   NewStr);
    }

    void idCharFilter(llvm::StringRef ref) {
        std::string str = ref.str();
        size_t size = ref.size();
        for (size_t i = 0; i < size; i++)
            if (!isalnum(str[i]) && str[i] != '_')
                str[i] = '_';
    }

public:
    RewriteCUDA(CompilerInstance *comp, llvm::raw_ostream *HostOS,
                llvm::raw_ostream *KernelOS) :
        ASTConsumer(), CI(comp),
        MainOutFile(HostOS), MainKernelOutFile(KernelOS) { }

    virtual ~RewriteCUDA() { }

    virtual void Initialize(ASTContext &Context) {
        SM = &Context.getSourceManager();
        PP = &CI->getPreprocessor();

        PP->addPPCallbacks(new RewriteIncludesCallback(this));

        HostRewrite.setSourceMgr(Context.getSourceManager(), Context.getLangOptions());
        KernelRewrite.setSourceMgr(Context.getSourceManager(), Context.getLangOptions());
        MainFileID = Context.getSourceManager().getMainFileID();

        HostIncludes += "#ifdef __APPLE__\n";
        HostIncludes += "#include <OpenCL/opencl.h>\n";
        HostIncludes += "#else\n";
        HostIncludes += "#include <CL/opencl.h>\n";
        HostIncludes += "#endif\n";
        HostIncludes += "#include <stdlib.h>\n";
        HostIncludes += "#include <stdio.h>\n";
        HostGlobalVars += "cl_platform_id clPlatform;\n";
        HostGlobalVars += "cl_device_id clDevice;\n";
        HostGlobalVars += "cl_context clContext;\n";
        HostGlobalVars += "cl_command_queue clCommandQueue;\n\n";
        HostGlobalVars += "size_t globalWorkSize[3];\n";
        HostGlobalVars += "size_t localWorkSize[3];\n";
        HostFunctions += "size_t loadProgramSource(char *filename, const char **progSrc) {\n" \
                        "\tFILE *f = fopen(filename, \"r\");\n" \
                        "\tfseek(f, 0, SEEK_END);\n" \
                        "\tsize_t len = (size_t) ftell(f);\n" \
                        "\t*progSrc = (const char *) malloc(sizeof(char)*len);\n" \
                        "\trewind(f);\n" \
                        "\tfread((void *) *progSrc, len, 1, f);\n" \
                        "\tfclose(f);\n" \
                        "\treturn len;\n" \
                        "}\n\n";

        IncludingStringH = false;
        UsesCUDADeviceProp = false;
        UsesCUDAMemset = false;
    }

    virtual void HandleTopLevelDecl(DeclGroupRef DG) {
        //Check where the declaration(s) comes from (may have been included)
        Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
        SourceLocation loc = firstDecl->getLocation();
        if (!SM->isFromMainFile(loc)) {
            if (strstr(SM->getPresumedLoc(loc).getFilename(), ".cu") != NULL) {
                if (OutFiles.find(SM->getFileID(loc)) == OutFiles.end()) {
                    //Create new files
                    FileID fileid = SM->getFileID(loc);
                    std::string filename = SM->getPresumedLoc(loc).getFilename();
                    size_t dotPos = filename.rfind('.');
                    filename = filename.substr(0, dotPos) + "-cl" + filename.substr(dotPos);
                    llvm::raw_ostream *hostOS = CI->createDefaultOutputFile(false, filename, "h");
                    llvm::raw_ostream *kernelOS = CI->createDefaultOutputFile(false, filename, "cl");
                    if (hostOS && kernelOS) {
                        OutFiles[fileid] = hostOS;
                        KernelOutFiles[fileid] = kernelOS;
                    }
                    else {
                        //TODO report and print error
                    }
                }
            }
            else {
                return;
            }
        }
        //Store VarDecl DeclGroupRefs
        if (firstDecl->getKind() == Decl::Var) {
            GlobalVarDeclGroups.insert(DG);
        }
        //Walk declarations in group and rewrite
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
                if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    RewriteKernelFunction(fd);
                }
                else {
                    RewriteHostFunction(fd);
                }
                if (fd->getNameAsString() == "main") {
                    RewriteMain(fd);
                }
            }
            else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                RemoveVar(vd, KernelRewrite);
                RewriteVarDecl(vd);
            }
        }
    }

    virtual void HandleTranslationUnit(ASTContext &) {
        //Declare global clPrograms
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            llvm::StringRef r = (*i).first;
            idCharFilter(r);
            HostGlobalVars += "cl_program clProgram_" + r.str() + ";\n";
        }
        //Insert host preamble at top of main file
        HostPreamble = HostIncludes + "\n" + HostDecls + "\n" + HostGlobalVars + "\n" + HostKernels + "\n" + HostFunctions;
        HostRewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), HostPreamble);
        //Insert device preamble at top of main kernel file
        DevPreamble = DevFunctions;
        KernelRewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), DevPreamble);

        CompoundStmt *mainBody = dyn_cast<CompoundStmt>(MainDecl->getBody());
        //Insert opencl initialization stuff at top of main
        CLInit += "\n";
        CLInit += "const char *progSrc;\n";
        CLInit += "size_t progLen;\n\n";
        CLInit += "clGetPlatformIDs(1, &clPlatform, NULL);\n";
        CLInit += "clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);\n";
        CLInit += "clContext = clCreateContext(NULL, 1, &clDevice, NULL, NULL, NULL);\n";
        CLInit += "clCommandQueue = clCreateCommandQueue(clContext, clDevice, 0, NULL);\n";
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            llvm::StringRef r = (*i).first;
            idCharFilter(r);
            CLInit += "progLen = loadProgramSource(\"" + r.str() + "-cl.cl\", &progSrc);\n";
            CLInit += "clProgram_" + r.str() + " = clCreateProgramWithSource(clContext, 1, &progSrc, &progLen, NULL);\n";
            CLInit += "free((void *) progSrc);\n";
            CLInit += "clBuildProgram(clProgram_" + r.str() + ", 1, &clDevice, \"-I .\", NULL, NULL);\n";
        }
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                llvm::StringRef r = (*i).first;
                idCharFilter(r);
                std::string kernelName = (*li).str();
                CLInit += "clKernel_" + kernelName + " = clCreateKernel(clProgram_" + r.str() + ", \"" + kernelName + "\", NULL);\n";
            }
        }
        HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(mainBody->getLBracLoc()), CLInit);

        //Insert cleanup code at bottom of main
        CLClean += "\n";
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLClean += "clReleaseKernel(clKernel_" + kernelName + ");\n";
            }
        }
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            llvm::StringRef r = (*i).first;
            idCharFilter(r);
            CLClean += "clReleaseProgram(clProgram_" + r.str() + ");\n";
        }
        CLClean += "clReleaseCommandQueue(clCommandQueue);\n";
        CLClean += "clReleaseContext(clContext);\n";
        HostRewrite.InsertTextBefore(mainBody->getRBracLoc(), CLClean);

        //Rewrite cl_mems in DeclGroups
        for (std::set<DeclGroupRef>::iterator i = DeviceMemDGs.begin(),
             e = DeviceMemDGs.end(); i != e; i++) {
            DeclGroupRef DG = *i;
            SourceLocation start, end;
            std::string replace;
            for (DeclGroupRef::iterator iDG = DG.begin(), eDG = DG.end(); iDG != eDG; ++iDG) {
                VarDecl *vd = (VarDecl *) (*iDG);
                if (iDG == DG.begin()) {
                    start = (*iDG)->getLocStart();
                }
                if ((iDG + 1) == DG.end()) {
                    end = (*iDG)->getLocEnd();
                }
                if (DeviceMemVars.find(vd) != DeviceMemVars.end()) {
                    //Change variable's type to cl_mem
                    replace += "cl_mem " + vd->getNameAsString() + ";\n";
                }
                else {
                    replace += PrintDeclToString(vd) + ";\n";
                }
            }
            HostRewrite.ReplaceText(start, HostRewrite.getRangeSize(SourceRange(start, end)), replace);
        }

        //Output main file's rewritten buffer
        if (const RewriteBuffer *RewriteBuff =
            HostRewrite.getRewriteBufferFor(MainFileID)) {
            *MainOutFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
        }
        else {
            //TODO use diagnostics for pretty errors
            llvm::errs() << "No changes made to " << SM->getFileEntryForID(MainFileID)->getName() << "\n";
        }
        //Output main kernel file's rewritten buffer
        if (const RewriteBuffer *RewriteBuff =
            KernelRewrite.getRewriteBufferFor(MainFileID)) {
            *MainKernelOutFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
        }
        else {
            //TODO use diagnostics for pretty errors
            llvm::errs() << "No changes made to " << SM->getFileEntryForID(MainFileID)->getName() << " kernel\n";
        }
        //Flush rewritten files
        MainOutFile->flush();
        MainKernelOutFile->flush();

        for (IDOutFileMap::iterator i = OutFiles.begin(), e = OutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
            if (const RewriteBuffer *RewriteBuff =
                HostRewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                //TODO use diagnostics for pretty errors
                llvm::errs() << "No changes made to " << SM->getFileEntryForID(fid)->getName() << "\n";
            }
            outFile->flush();
        }
        for (IDOutFileMap::iterator i = KernelOutFiles.begin(), e = KernelOutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
            if (const RewriteBuffer *RewriteBuff =
                KernelRewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                //TODO use diagnostics for pretty errors
                llvm::errs() << "No changes made to " << SM->getFileEntryForID(fid)->getName() << " kernel\n";
            }
            outFile->flush();
        }
    }

    void RewriteInclude(SourceLocation HashLoc, const Token &IncludeTok,
                        llvm::StringRef FileName, bool IsAngled,
                        const FileEntry *File, SourceLocation EndLoc/*,
                        const llvm::SmallVectorImpl<char> &RawPath*/) {
        if (SM->isFromMainFile(HashLoc) ||
            extension(SM->getPresumedLoc(HashLoc).getFilename()).str() == ".cu") {
            if (IsAngled) {
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                if (filename(FileName).str() == "cuda.h")
                    HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
            else if (filename(FileName).str() == "cuda.h") {
                HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
            else if (extension(FileName).str() == ".cu") {
                FileID fileID = SM->getFileID(HashLoc);
                SourceLocation fileStartLoc = SM->getLocForStartOfFile(fileID);
                llvm::StringRef fileBuf = SM->getBufferData(fileID);
                const char *fileBufStart = fileBuf.begin();

                llvm::StringRef ext = extension(FileName);
                SourceLocation start = fileStartLoc.getFileLocWithOffset(ext.begin() - fileBufStart);
                SourceLocation end = fileStartLoc.getFileLocWithOffset((ext.end()-1) - fileBufStart);
                HostRewrite.ReplaceText(start, HostRewrite.getRangeSize(SourceRange(start, end)), "-cl.h");
                KernelRewrite.ReplaceText(start, KernelRewrite.getRangeSize(SourceRange(start, end)), "-cl.cl");
            }
            else {
                //TODO store include info to rewrite later?
            }
        }
    }

};

class RewriteCUDAAction : public PluginASTAction {
protected:
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) {
        std::string filename = InFile.str();
        size_t dotPos = filename.rfind('.');
        filename = filename.substr(0, dotPos) + "-cl" + filename.substr(dotPos);
        llvm::raw_ostream *hostOS = CI.createDefaultOutputFile(false, filename, "cpp");
        llvm::raw_ostream *kernelOS = CI.createDefaultOutputFile(false, filename, "cl");
        if (hostOS && kernelOS)
            return new RewriteCUDA(&CI, hostOS, kernelOS);
        //TODO cleanup files?
        return NULL;
    }

    bool ParseArgs(const CompilerInstance &CI,
                   const std::vector<std::string> &args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
            llvm::errs() << "RewriteCUDA arg = " << args[i] << "\n";
            //TODO parse arguments
        }
        if (args.size() && args[0] == "help")
            PrintHelp(llvm::errs());

        return true;
    }

    void PrintHelp(llvm::raw_ostream &ros) {
        ros << "Help for RewriteCUDA plugin goes here\n";
    }

};

RewriteIncludesCallback::RewriteIncludesCallback(RewriteCUDA *RC) :
    RCUDA(RC) {
}

void RewriteIncludesCallback::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                                                 llvm::StringRef FileName, bool IsAngled,
                                                 const FileEntry *File, SourceLocation EndLoc/*,
                                                 const llvm::SmallVectorImpl<char> &RawPath*/) {
    RCUDA->RewriteInclude(HashLoc, IncludeTok, FileName, IsAngled, File, EndLoc);
}

}

static FrontendPluginRegistry::Add<RewriteCUDAAction>
X("rewrite-cuda", "translate CUDA kernels to OpenCL");
