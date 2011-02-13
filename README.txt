This is a clang plugin for rewriting CUDA to OpenCL.

You will need to have a version of clang built with support for CUDA. Then, to
build the plugin, simply drop the entire directory into clang's examples
folder.

Once the plugin is built, you can run it using:
--
$ clang -cc1 -load path/to/RewriteCUDA.so -plugin rewrite-cuda some-input-file.cu
--
