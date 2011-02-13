//===- PrintFunctionNames.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which simply prints the names of all the top-level decls
// in the input file.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

#include "clang/Basic/SourceManager.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "clang/Rewrite/Rewriter.h"

#include "llvm/Support/raw_ostream.h"

#include <set>
//#include <sstream>
#include <string>

using namespace clang;

namespace {

/* Find all __global__ and __device__ functions and translate them
 * to OpenCL. */
class RewriteCUDA : public ASTConsumer {
private:
    SourceManager *SM;
    Rewriter Rewrite;
    FileID MainFileID;

    //TODO keep track of kernels available

    llvm::raw_ostream *HostOutFile;
    llvm::raw_ostream *KernelOutFile;

    std::set<llvm::StringRef> Kernels;

    std::string Preamble;

    void TraverseStmt(Stmt *e, unsigned int indent) {
        for (unsigned int i = 0; i < indent; i++)
            llvm::errs() << "  ";
        llvm::errs() << e->getStmtClassName() << "\n";
        indent++;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end(); CI != CE; ++CI)
            TraverseStmt(*CI, indent);
    }

    DeclRefExpr *FindDeclRefExpr(Stmt *e) {
        if (DeclRefExpr *dr = dyn_cast<DeclRefExpr>(e))
            return dr;
        DeclRefExpr *ret = NULL;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end(); CI != CE; ++CI) {
            ret = FindDeclRefExpr(*CI);
            if (ret)
                return ret;
        }
        return NULL;

    }

    void RewriteStmt(Stmt *s) {
        //TODO for recursively going through Stmts
        /*if (CallExpr *ce = dyn_cast<CallExpr>(body))
        {
            
        }*/
        /*switch (body->getStmtClass())
        {
        }*/
    }

    void RewriteCUDAKernel(FunctionDecl *cudaKernel) {
        llvm::errs() << "Rewriting CUDA kernel\n";
        Kernels.insert(cudaKernel->getName());
        Preamble += "cl_kernel clKernel_" + cudaKernel->getName().str() + ";\n";
        //TODO find way to rewrite attributes
        //GlobalAttr *ga = fd->getAttr<GlobalAttr>();
        //Rewrite.ReplaceText(ga->getLocation(), sizeof(char)*(sizeof("__global__")-1), "__kernel");
    }

    void RewriteCUDACall(CallExpr *cudaCall) {
        llvm::errs() << "Rewriting CUDA API call\n";
        FunctionDecl *cudaFunc = cudaCall->getDirectCallee();
        std::string funcName = cudaFunc->getNameAsString();
        if (funcName == "cudaThreadExit") {
            //Replace with clReleaseContext()
            Rewrite.ReplaceText(cudaCall->getExprLoc(),
                                Rewrite.getRangeSize(cudaCall->getSourceRange()),
                                "clReleaseContext(clContext)");
        }
        else if (funcName == "cudaThreadSynchronize") {
            //Replace with clFinish
            Rewrite.ReplaceText(cudaCall->getExprLoc(),
                                Rewrite.getRangeSize(cudaCall->getSourceRange()),
                                "clFinish(clCommandQueue)");
        }
        else if (funcName == "cudaMalloc") {
            //TODO check if the return value is being placed somewhere
            //TODO case if cudaMalloc being used as argument to something
            //TODO get arguments
            Expr *varExpr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            //TraverseStmt(varExpr, 0);
            //TraverseStmt(size, 0);
            DeclRefExpr *dr = FindDeclRefExpr(varExpr);
            llvm::StringRef varName = dr->getDecl()->getName();
            //llvm::errs() << "cudaMalloc var name: " << varName.str() << "\n";
            //TODO use printPretty printing to an llvm::raw_string_ostream to print AST portions
            std::string SStr;
            llvm::raw_string_ostream S(SStr);
            size->printPretty(S, 0, PrintingPolicy(Rewrite.getLangOpts()));
            //Replace with clCreateBuffer
            Rewrite.ReplaceText(cudaCall->getExprLoc(),
                                Rewrite.getRangeSize(cudaCall->getSourceRange()),
                                varName.str() + " = clCreateBuffer(clContext, CL_MEM_READ_WRITE, " + S.str() + ", NULL, NULL)");
        }
        else if (funcName == "cudaFree") {
            //Replace with clReleaseMemObject
            Rewrite.ReplaceText(cudaCall->getExprLoc(),
                                Rewrite.getRangeSize(cudaCall->getSourceRange()),
                                "clReleaseMemObject()");
        }
        else if (funcName == "cudaMemcpy") {
        }
        else if (funcName == "cudaMemset") {
        }
        else {
            llvm::errs() << "Unsupported CUDA call: " << funcName << "\n";
        }
    }

    void RewriteCUDAKernelCall(CUDAKernelCallExpr *kernelCall) {
        llvm::errs() << "Rewriting CUDA kernel call\n";
    }

public:
    RewriteCUDA(llvm::raw_ostream *HostOS, llvm::raw_ostream *KernelOS) :
        ASTConsumer(), HostOutFile(HostOS), KernelOutFile(KernelOS) { }

    virtual ~RewriteCUDA() { }

    virtual void Initialize(ASTContext &Context) {
        SM = &Context.getSourceManager();
        Rewrite.setSourceMgr(Context.getSourceManager(), Context.getLangOptions());
        MainFileID = Context.getSourceManager().getMainFileID();

        Preamble += "cl_platform_id clPlatform;\n";
        Preamble += "cl_device_id clDevice;\n";
        Preamble += "cl_context clContext;\n";
        Preamble += "cl_command_queue clCommandQueue;\n";
        Preamble += "cl_program clProgram;\n\n";
    }

    virtual void HandleTopLevelDecl(DeclGroupRef DG) {
        for(DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
                if (fd->hasAttr<GlobalAttr>() || fd->hasAttr<DeviceAttr>()) {
                    //Is a device function
                    RewriteCUDAKernel(fd);
                }
                else if (Stmt *body = fd->getBody()) {
                    assert(body->getStmtClass() == Stmt::CompoundStmtClass && "Invalid statement: Not a statement class");
                    CompoundStmt *cs = dyn_cast<CompoundStmt>(body);
                    llvm::errs() << "Number of Stmts: " << cs->size() << "\n";
                    for (Stmt::child_iterator ci = cs->child_begin(), ce = cs->child_end();
                         ci != ce; ++ci) {
                        if (Stmt *childStmt = *ci) {
                            llvm::errs() << "Child Stmt: " << childStmt->getStmtClassName() << "\n";
                            if (CallExpr *ce = dyn_cast<CallExpr>(childStmt)) {
                                llvm::errs() << "\tCallExpr: ";
                                if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(ce)) {
                                    RewriteCUDAKernelCall(kce);
                                }
                                if (FunctionDecl *callee = ce->getDirectCallee()) {
                                    llvm::errs() << callee->getName() << "\n";
                                    if (callee->getNameAsString().find("cuda") == 0) {
                                        RewriteCUDACall(ce);
                                    }
                                }
                                else
                                    llvm::errs() << "??\n";
                            }
                        }
                    }
                }
            }
        }
    }

    virtual void HandleTranslationUnit(ASTContext &) {
        //TODO add global CL declarations
        Rewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), Preamble);

        //Write the rewritten buffer to a file
        if (const RewriteBuffer *RewriteBuff =
            Rewrite.getRewriteBufferFor(MainFileID)) {
            *HostOutFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
        }
        else {
            llvm::errs() << "No changes made!\n";
        }
        HostOutFile->flush();
    }

};

class RewriteCUDAAction : public PluginASTAction {
protected:
  ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) {
    if (llvm::raw_ostream *HostOS = CI.createDefaultOutputFile(false, InFile, "c")) {
        if (llvm::raw_ostream *KernelOS = CI.createDefaultOutputFile(false, InFile, "cl")) {
            return new RewriteCUDA(HostOS, KernelOS);
        }
        //TODO cleanup HostOS
    }
    return NULL;
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string>& args) {
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      llvm::errs() << "RewriteCUDA arg = " << args[i] << "\n";

      // Example error handling.
      if (args[i] == "-an-error") {
        Diagnostic &D = CI.getDiagnostics();
        unsigned DiagID = D.getCustomDiagID(
          Diagnostic::Error, "invalid argument '" + args[i] + "'");
        D.Report(DiagID);
        return false;
      }
    }
    if (args.size() && args[0] == "help")
      PrintHelp(llvm::errs());

    return true;
  }
  void PrintHelp(llvm::raw_ostream& ros) {
    ros << "Help for RewriteCUDA plugin goes here\n";
  }

};

}

static FrontendPluginRegistry::Add<RewriteCUDAAction>
X("rewrite-cuda", "translate CUDA kernels to OpenCL");
