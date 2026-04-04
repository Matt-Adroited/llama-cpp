// TurboQuant tq3_0 K / tq2_0 V flash attention template instances

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TQ3_0, GGML_TYPE_TQ2_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ3_0, GGML_TYPE_TQ2_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ3_0, GGML_TYPE_TQ2_0);
