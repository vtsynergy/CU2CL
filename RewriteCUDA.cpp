#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "clang/Lex/Preprocessor.h"

#include "clang/Rewrite/Rewriter.h"

#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <sstream>
#include <string>

using namespace clang;

namespace {

/**
 * An AST consumer made to rewrite CUDA to OpenCL.
 **/
class RewriteCUDA : public ASTConsumer {
private:
    typedef std::map<FileID, llvm::raw_ostream *> IDOutFileMap;
    typedef std::pair<FileID, llvm::raw_ostream *> IDOutFilePair;

    CompilerInstance *CI;
    SourceManager *SM;
    Preprocessor *PP;
    Rewriter Rewrite;

    //Rewritten files
    FileID MainFileID;
    llvm::raw_ostream *MainOutFile;
    llvm::raw_ostream *MainKernelOutFile;
    IDOutFileMap OutFiles;
    IDOutFileMap KernelOutFiles;

    std::set<llvm::StringRef> Kernels;
    std::set<VarDecl *> DeviceMems;

    FunctionDecl *MainDecl;

    std::string Preamble;
    //TODO break Preamble up into different portions that are combined

    std::string CLInit;
    std::string CLClean;

    bool IncludingStringH;

    void TraverseStmt(Stmt *e, unsigned int indent) {
        for (unsigned int i = 0; i < indent; i++)
            llvm::errs() << "  ";
        llvm::errs() << e->getStmtClassName() << "\n";
        indent++;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI)
            TraverseStmt(*CI, indent);
    }

    //TODO include template?
    DeclRefExpr *FindDeclRefExpr(Stmt *e) {
        if (DeclRefExpr *dr = dyn_cast<DeclRefExpr>(e))
            return dr;
        DeclRefExpr *ret = NULL;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI) {
            ret = FindDeclRefExpr(*CI);
            if (ret)
                return ret;
        }
        return NULL;

    }

    void RewriteHostStmt(Stmt *s) {
        //Traverse children and recurse
        for (Stmt::child_iterator CI = s->child_begin(), CE = s->child_end();
             CI != CE; ++CI) {
            if (*CI)
                RewriteHostStmt(*CI);
        }
        //Visit this node
        //TODO support for, while, do-while, if, CompoundStmts
        if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(s)) {
            RewriteCUDAKernelCall(kce);
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(s)) {
            if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0)
                RewriteCUDACall(ce);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
            DeclGroupRef DG = ds->getDeclGroup();
            for(DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    RewriteVarDecl(vd);
                }
                //TODO other non-top level declarations??
            }
        }
    }

    void RewriteCUDAKernel(FunctionDecl *cudaKernel) {
        //llvm::errs() << "Rewriting CUDA kernel\n";
        Kernels.insert(cudaKernel->getName());
        Preamble += "cl_kernel clKernel_" + cudaKernel->getName().str() + ";\n";
        //Rewrite kernel attributes
        if (cudaKernel->hasAttr<CUDAGlobalAttr>()) {
            SourceLocation loc = FindAttr(cudaKernel->getLocStart(), "__global__");
            RewriteAttr("__global__", loc);
        }
        if (cudaKernel->hasAttr<CUDADeviceAttr>()) {
            SourceLocation loc = FindAttr(cudaKernel->getLocStart(), "__device__");
            RewriteAttr("__device__", loc);
        }
        if (cudaKernel->hasAttr<CUDAHostAttr>()) {
            SourceLocation loc = FindAttr(cudaKernel->getLocStart(), "__host__");
            RewriteAttr("__host__", loc);
            //TODO leave rewritten code in original file
        }
        //Rewrite arguments
        for (FunctionDecl::param_iterator PI = cudaKernel->param_begin(),
                                          PE = cudaKernel->param_end();
                                          PI != PE; ++PI) {
            RewriteKernelParam(*PI);
        }
        //Rewirte kernel body
        if (cudaKernel->hasBody()) {
            //TraverseStmt(cudaKernel->getBody(), 0);
            RewriteKernelStmt(cudaKernel->getBody());
        }
        //TODO remove kernel from original source location
    }

    void RewriteKernelParam(ParmVarDecl *parmDecl) {
        if (parmDecl->hasAttr<CUDADeviceAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__device__");
            RewriteAttr("__device__", loc);
        }
        else if (parmDecl->hasAttr<CUDAConstantAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__constant__");
            RewriteAttr("__constant__", loc);
        }
        else if (parmDecl->hasAttr<CUDASharedAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__shared__");
            RewriteAttr("__shared__", loc);
        }
        else if (parmDecl->getOriginalType().getTypePtr()->isPointerType()) {
            Rewrite.InsertTextBefore(
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
            str = PrintPrettyToString(me);
            if (str == "threadIdx.x")
                ReplaceStmtWithText(ks, "get_local_id(0)");
            else if (str == "threadIdx.y")
                ReplaceStmtWithText(ks, "get_local_id(1)");
            else if (str == "threadIdx.z")
                ReplaceStmtWithText(ks, "get_local_id(2)");
            else if (str == "blockIdx.x")
                ReplaceStmtWithText(ks, "get_group_id(0)");
            else if (str == "blockIdx.y")
                ReplaceStmtWithText(ks, "get_group_id(1)");
            else if (str == "blockIdx.z")
                ReplaceStmtWithText(ks, "get_group_id(2)");
            else if (str == "blockDim.x")
                ReplaceStmtWithText(ks, "get_local_size(0)");
            else if (str == "blockDim.y")
                ReplaceStmtWithText(ks, "get_local_size(1)");
            else if (str == "blockDim.z")
                ReplaceStmtWithText(ks, "get_local_size(2)");
            else if (str == "gridDim.x")
                ReplaceStmtWithText(ks, "get_group_size(0)");
            else if (str == "gridDim.y")
                ReplaceStmtWithText(ks, "get_group_size(1)");
            else if (str == "gridDim.z")
                ReplaceStmtWithText(ks, "get_group_size(2)");
        }
        /*else if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(ks)) {
        }*/
        else if (CallExpr *ce = dyn_cast<CallExpr>(ks)) {
            str = PrintPrettyToString(ce);
            if (str == "__syncthreads()")
                ReplaceStmtWithText(ks, "barrier(CLK_LOCAL_MEM_FENCE)");
        }
    }

    void RewriteCUDACall(CallExpr *cudaCall) {
        //llvm::errs() << "Rewriting CUDA API call\n";
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();
        if (funcName == "cudaThreadExit") {
            //Replace with clReleaseContext()
            ReplaceStmtWithText(cudaCall, "clReleaseContext(clContext)");
        }
        else if (funcName == "cudaThreadSynchronize") {
            //Replace with clFinish
            ReplaceStmtWithText(cudaCall, "clFinish(clCommandQueue)");
        }
        else if (funcName == "cudaSetDevice") {
            //TODO implement
            llvm::errs() << "cudaSetDevice not implemented yet\n";
        }
        else if (funcName == "cudaMalloc") {
            //TODO check if the return value is being used somehow?
            Expr *varExpr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            DeclRefExpr *dr = FindDeclRefExpr(varExpr);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            llvm::StringRef varName = var->getName();
            std::string str;
            str = PrintPrettyToString(size);

            //Replace with clCreateBuffer
            std::string sub = varName.str() + " = clCreateBuffer(clContext, CL_MEM_READ_WRITE, " + str + ", NULL, NULL)";
            ReplaceStmtWithText(cudaCall, sub);

            //Change variable's type to cl_mem
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            Rewrite.ReplaceText(tl.getBeginLoc(),
                                Rewrite.getRangeSize(tl.getSourceRange()),
                                "cl_mem ");

            //Add var to DeviceMems
            DeviceMems.insert(var);
        }
        else if (funcName == "cudaFree") {
            Expr *devPtr = cudaCall->getArg(0);
            DeclRefExpr *dr = FindDeclRefExpr(devPtr);
            llvm::StringRef varName = dr->getDecl()->getName();

            //Replace with clReleaseMemObject
            ReplaceStmtWithText(cudaCall, "clReleaseMemObject(" + varName.str() + ")");
        }
        else if (funcName == "cudaMemcpy") {
            std::string replace;

            Expr *dst = cudaCall->getArg(0);
            Expr *src = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            Expr *kind = cudaCall->getArg(3);
            DeclRefExpr *dr = FindDeclRefExpr(kind);
            EnumConstantDecl *enumConst = dyn_cast<EnumConstantDecl>(dr->getDecl());
            std::string enumString = enumConst->getNameAsString();
            //llvm::errs() << enumString << "\n";
            if (enumString == "cudaMemcpyHostToHost") {
                //standard memcpy
                //make sure to include <string.h>
                if (!IncludingStringH) {
                    Preamble = "#include <string.h>\n\n" + Preamble;
                    IncludingStringH = true;
                }
                replace += "memcpy(";
                replace += PrintPrettyToString(dst) + ", ";
                replace += PrintPrettyToString(src) + ", ";
                replace += PrintPrettyToString(count) + ")";
                ReplaceStmtWithText(cudaCall, replace);
            }
            else if (enumString == "cudaMemcpyHostToDevice") {
                //clEnqueueWriteBuffer
                replace += "clEnqueueWriteBuffer(clCommandQueue, ";
                replace += PrintPrettyToString(dst) + ", CL_TRUE, 0, ";
                replace += PrintPrettyToString(count) + ", ";
                replace += PrintPrettyToString(src) + ", 0, NULL, NULL)";
                ReplaceStmtWithText(cudaCall, replace);
            }
            else if (enumString == "cudaMemcpyDeviceToHost") {
                //clEnqueueReadBuffer
                replace += "clEnqueueReadBuffer(clCommandQueue, ";
                replace += PrintPrettyToString(src) + ", CL_TRUE, 0, ";
                replace += PrintPrettyToString(count) + ", ";
                replace += PrintPrettyToString(dst) + ", 0, NULL, NULL)";
                ReplaceStmtWithText(cudaCall, replace);
            }
            else if (enumString == "cudaMemcpyDeviceToDevice") {
                //TODO clEnqueueReadBuffer; clEnqueueWriteBuffer
                ReplaceStmtWithText(cudaCall, "clEnqueueReadBuffer(clCommandQueue, src, CL_TRUE, 0, count, temp, 0, NULL, NULL)");
                ReplaceStmtWithText(cudaCall, "clEnqueueWriteBuffer(clCommandQueue, dst, CL_TRUE, 0, count, temp, 0, NULL, NULL)");
            }
            else {
                //TODO Use diagnostics to print pretty errors
                llvm::errs() << "Unsupported cudaMemcpy type: " << enumString << "\n";
            }
        }
        else if (funcName == "cudaMemset") {
            //TODO follow Swan's example of setting via a kernel
            //TODO add memset kernel to the list of kernels
            /*Expr *devPtr = cudaCall->getArg(0);
            Expr *value = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);*/
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
        for (unsigned i = 0; i < kernelCall->getNumArgs(); i++) {
            Expr *arg = kernelCall->getArg(i);
            VarDecl *var = dyn_cast<VarDecl>(FindDeclRefExpr(arg)->getDecl());
            args << "clSetKernelArg(" << kernelName << ", " << i << ", sizeof(";
            if (DeviceMems.find(var) != DeviceMems.end()) {
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
                for (unsigned int i = 0; i < 3; i++)
                    args << "localWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "];\n";
            }
            else {
                //TODO ??
            }
        }
        else {
            //constant passed to block
            Expr *arg = cast->getSubExprAsWritten();
            args << "localWorkSize[0] = " << PrintPrettyToString(arg) << ";\n";
        }
        construct = dyn_cast<CXXConstructExpr>(grid);
        cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
            //variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                for (unsigned int i = 0; i < 3; i++)
                    args << "globalWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "]*localWorkSize[" << i << "];\n";
            }
            else {
                //TODO ??
            }
        }
        else {
            //constant passed to grid
            Expr *arg = cast->getSubExprAsWritten();
            args << "globalWorkSize[0] = (" << PrintPrettyToString(arg) << ")*localWorkSize[0];\n";
        }
        args << "clEnqueueNDRangeKernel(clCommandQueue, " << kernelName << ", 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)";
        ReplaceStmtWithText(kernelCall, args.str());
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
            Rewrite.ReplaceText(tl.getBeginLoc(),
                                Rewrite.getRangeSize(tl.getSourceRange()),
                                "size_t");
            if (var->hasInit()) {
                //Rewrite initial value
                CXXConstructExpr *ce = (CXXConstructExpr *) var->getInit();
                std::string args = " = {";
                for (CXXConstructExpr::arg_iterator i = ce->arg_begin(), e = ce->arg_end();
                     i != e; ++i) {
                    Expr *arg = *i;
                    if (CXXDefaultArgExpr *defArg = dyn_cast<CXXDefaultArgExpr>(arg)) {
                        args += PrintPrettyToString(defArg->getExpr());
                    }
                    else {
                        args += PrintPrettyToString(arg);
                    }
                    if (i + 1 != e)
                        args += ", ";
                }
                args += "}";
                SourceRange parenRange = ce->getParenRange();
                if (parenRange.isValid()) {
                    Rewrite.ReplaceText(parenRange.getBegin(),
                                        Rewrite.getRangeSize(parenRange),
                                        args);
                    Rewrite.InsertTextBefore(parenRange.getBegin(), "[3]");
                }
                else {
                    Rewrite.InsertTextAfter(PP->getLocForEndOfToken(var->getLocEnd()), "[3]");
                    Rewrite.InsertTextAfter(PP->getLocForEndOfToken(var->getLocEnd()),
                                            args);
                }
            }
        }
        //TODO check other CUDA-only types to rewrite
    }

    void RewriteAttr(std::string attr, SourceLocation loc) {
        //llvm::StringRef fileBuf = SM->getBufferData(fileID);
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

    std::string PrintPrettyToString(Stmt *s) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        s->printPretty(S, 0, PrintingPolicy(Rewrite.getLangOpts()));
        return S.str();
    }

    bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr) {
        return Rewrite.ReplaceText(OldStmt->getLocStart(),
                                   Rewrite.getRangeSize(OldStmt->getSourceRange()),
                                   NewStr);
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
        Rewrite.setSourceMgr(Context.getSourceManager(), Context.getLangOptions());
        MainFileID = Context.getSourceManager().getMainFileID();

        Preamble += "#ifdef __APPLE__\n";
        Preamble += "#include <OpenCL/opencl.h>\n";
        Preamble += "#else\n";
        Preamble += "#include <CL/opencl.h>\n";
        Preamble += "#endif\n\n";
        Preamble += "cl_platform_id clPlatform;\n";
        Preamble += "cl_device_id clDevice;\n";
        Preamble += "cl_context clContext;\n";
        Preamble += "cl_command_queue clCommandQueue;\n";
        Preamble += "cl_program clProgram;\n\n";
        Preamble += "size_t globalWorkSize[3];\n";
        Preamble += "size_t localWorkSize[3];\n\n";

        IncludingStringH = false;
    }

    virtual void HandleTopLevelDecl(DeclGroupRef DG) {
        //Check where the declaration(s) comes from (may have been included)
        if (!DG.isNull()) {
            SourceLocation loc = (DG.isSingleDecl() ? DG.getSingleDecl() :
                                  DG.getDeclGroup()[0])->getLocation();
            if (strstr(SM->getPresumedLoc(loc).getFilename(), ".cu") != NULL) {
                if (OutFiles.find(SM->getFileID(loc)) == OutFiles.end()) {
                    //Create new files
                    FileID fileid = SM->getFileID(loc);
                    std::string filename = SM->getPresumedLoc(loc).getFilename();
                    size_t dotPos = filename.rfind('.');
                    filename = filename.substr(0, dotPos) + "-cl" + filename.substr(dotPos);
                    llvm::raw_ostream *hostOS = CI->createDefaultOutputFile(false, filename, "cpp");
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
            else if (!SM->isFromMainFile(loc)) {
                return;
            }
            for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                        //Is a device function
                        RewriteCUDAKernel(fd);
                    }
                    else if (Stmt *body = fd->getBody()) {
                        RewriteHostStmt(body);
                    }
                    if (fd->getNameAsString() == "main") {
                        RewriteMain(fd);
                    }
                }
                else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    RewriteVarDecl(vd);
                }
            }
        }
    }

    virtual void HandleTranslationUnit(ASTContext &) {
        //Add global CL declarations
        Rewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), Preamble);

        CompoundStmt *mainBody = dyn_cast<CompoundStmt>(MainDecl->getBody());
        //Add opencl initialization stuff at top of main
        CLInit += "\n";
        CLInit += "clGetPlatformIDs(1, &clPlatform, NULL);\n";
        CLInit += "clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);\n";
        CLInit += "clContext = clCreateContext(NULL, 1, &clDevice, NULL, NULL, NULL);\n";
        CLInit += "clCommandQueue = clCreateCommandQueue(clContext, clDevice, 0, NULL);\n";
        CLInit += "clProgram = clCreateProgramWithSource(clContext, count, strings, lengths, NULL);\n";
        CLInit += "clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);\n";
        for (std::set<llvm::StringRef>::iterator it = Kernels.begin();
             it != Kernels.end(); it++) {
            std::string kernelName = (*it).str();
            CLInit += "clKernel_" + kernelName + " = clCreateKernel(clProgram, \"" + kernelName + "\", NULL);\n";
        }
        Rewrite.InsertTextAfter(PP->getLocForEndOfToken(mainBody->getLBracLoc()), CLInit);

        //Add cleanup code at bottom of main
        CLClean += "\n";
        for (std::set<llvm::StringRef>::iterator it = Kernels.begin();
             it != Kernels.end(); it++) {
            std::string kernelName = (*it).str();
            CLClean += "clReleaseKernel(clKernel_" + kernelName + ");\n";
        }
        CLClean += "clReleaseProgram(clProgram);\n";
        CLClean += "clReleaseCommandQueue(clCommandQueue);\n";
        CLClean += "clReleaseContext(clContext);\n";
        Rewrite.InsertTextBefore(mainBody->getRBracLoc(), CLClean);

        //Write the rewritten buffer to a file
        if (const RewriteBuffer *RewriteBuff =
            Rewrite.getRewriteBufferFor(MainFileID)) {
            *MainOutFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
        }
        else {
            //TODO use diagnostics for pretty errors
            llvm::errs() << "No changes made!\n";
        }
        //TODO Write the rewritten kernel buffers to new files

        //Flush rewritten files
        MainOutFile->flush();
        MainKernelOutFile->flush();
        for (IDOutFileMap::iterator i = OutFiles.begin(), e = OutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
            if (const RewriteBuffer *RewriteBuff =
                Rewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                //TODO use diagnostics for pretty errors
                llvm::errs() << "No changes made!\n";
            }
            outFile->flush();
        }
        for (IDOutFileMap::iterator i = KernelOutFiles.begin(), e = KernelOutFiles.end();
             i != e; i++) {
            (*i).second->flush();
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

}

static FrontendPluginRegistry::Add<RewriteCUDAAction>
X("rewrite-cuda", "translate CUDA kernels to OpenCL");
