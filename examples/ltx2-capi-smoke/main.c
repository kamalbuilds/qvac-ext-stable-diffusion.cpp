/* Link/ABI smoke test for the LTX-2 C API: no model weights required.
 * Checks that ltx2_new_ctx() fails cleanly (returns NULL, no crash) when the
 * diffusion model path does not exist. Uses an explicit check rather than
 * assert() so the test still fires in a Release (-DNDEBUG) build. */
#include <stdio.h>
#include <stdlib.h>

#include "ltx2.h"

int main(void) {
    ltx2_ctx_t* ctx = ltx2_new_ctx("/nonexistent/diffusion_model.gguf",
                                    "/nonexistent/vae.safetensors",
                                    "/nonexistent/gemma.gguf",
                                    /*n_threads=*/1,
                                    SD_TYPE_F16);

    if (ctx != NULL) {
        fprintf(stderr, "ltx2-capi-smoke: FAIL (ltx2_new_ctx returned non-NULL for missing models)\n");
        ltx2_free_ctx(ctx);
        return 1;
    }

    ltx2_free_ctx(ctx); /* no-op on NULL; also exercises the NULL-safe path */

    printf("ltx2-capi-smoke: PASS (ltx2_new_ctx rejected missing model files)\n");
    return 0;
}
