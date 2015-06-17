//Global configuration for each of the test codes that has a "config*" method

//For now, most share a global configuration, but this is built to be changed..
#define GLOBAL_GRID 128
#define GLOBAL_BLOCK 512
#define GLOBAL_COUNT 65536

//kernel/pointer_qualifiers.cu
#define PQT_grid GLOBAL_GRID
#define PQT_block GLOBAL_BLOCK
#define PQT_count GLOBAL_COUNT

//kernel/struct_params.cu
#define TSP_grid GLOBAL_GRID
#define TSP_block GLOBAL_BLOCK
#define TSP_count GLOBAL_COUNT
