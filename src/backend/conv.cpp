#include <torch/extension.h>

uint32_t get_icosph_level(uint32_t V);

static inline uint32_t wrap(uint32_t value, uint32_t T)
{
    return value >= T ? value - T : value;
}

torch::Tensor icosphere_gradient_cpu(torch::Tensor input, torch::Tensor weight_table, bool include_value)
{
    TORCH_CHECK(input.dim() == 3, "Input must be (N, C_in, V)");

    uint32_t N = input.size(0);  // batch size
    uint32_t C = input.size(1);  // channel size
    uint32_t V = input.size(2);  // vertex count
    uint32_t L = get_icosph_level(V);  // icosphere level

    input = input.contiguous();
    const float* input_ptr = input.data_ptr<float>();

    TORCH_CHECK(weight_table.is_contiguous(), "Neighbour weight table must be contiguous");
    const float* weight_ptr = weight_table.data_ptr<float>();

    uint32_t G = include_value ? 5 : 4;   // value per vertex/channel count
    torch::Tensor output = torch::zeros({N, C, V, G}, input.options());
    float* out_ptr = output.data_ptr<float>();

    uint32_t n_face_rings = 1 << L;     // how many vertex rings per face side
    uint32_t half_ring_size = 2 * n_face_rings; // how many vertices per half ring

    int32_t base_off[6] {V-half_ring_size-1, half_ring_size+1, V-half_ring_size, 1, V-1, half_ring_size};
    int32_t east_off[6] = {V-n_face_rings-1, half_ring_size+1, V-n_face_rings, 1, V-1, half_ring_size};

    for (uint32_t n = 0; n < N; n++)
    {
        float* out_ptr = out_ptr + n * (C * V * G);
        const float* in_ptr = in_ptr + n * (C * V);

        for (uint32_t c = 0; c < C; c++)
        {
            float* out_ptr = out_ptr + c * (V * G);
            const float* in_ptr = in_ptr + c * V;

            for (uint32_t f = 0; f < 5; f++)
            {
                uint32_t face_0 = f * (half_ring_size * n_face_rings);

                // middle
                for (uint32_t j = 1; j < n_face_rings - 1; j++)
                {
                    uint32_t ring_0 = face_0 + j * half_ring_size;
                    for (uint32_t i = 1; i < half_ring_size - 1; i++)
                    {
                        uint32_t v_index = ring_0 + i;
                        const float* w = weight_ptr + 6 * v_index;
                        float* out = out_ptr + v_index * G;
                        
                        float value = in_ptr[v_index];
                        out[0] = w[0] * (in_ptr[wrap(v_index + base_off[0], V)] - value);
                        out[1] = w[1] * (in_ptr[wrap(v_index + base_off[1], V)] - value);
                        out[2] = w[2] * (in_ptr[wrap(v_index + base_off[2], V)] - value) +
                                 w[3] * (in_ptr[wrap(v_index + base_off[3], V)] - value);
                        out[3] = w[4] * (in_ptr[wrap(v_index + base_off[4], V)] - value) + 
                                 w[5] * (in_ptr[wrap(v_index + base_off[5], V)] - value);
                    }
                }
            
                // east seam
                for (uint32_t i = 1; i < half_ring_size - n_face_rings; i++)
                {
                    uin32_t v_index = face_0 + i;
                    const float* w = weight_ptr + 6 * v_index;
                    float* out = out_ptr + v_index * G;
                    
                    float value = in_ptr[v_index];
                    out[0] = w[0] * (in_ptr[wrap(v_index + east_off[0], V)] - value);
                    out[1] = w[1] * (in_ptr[wrap(v_index + east_off[1], V)] - value);
                    out[2] = w[2] * (in_ptr[wrap(v_index + east_off[2], V)] - value) + 
                             w[3] * (in_ptr[wrap(v_index + east_off[3], V)] - value);
                    out[3] = w[4] * (in_ptr[wrap(v_index + east_off[4], V)] - value) + 
                             w[5] * (in_ptr[wrap(v_index + east_off[5], V)] - value);
                    }
                }

                // north east seam
                for (uint32_t i = half_ring_size - n_face_rings; i < half_ring_size - 1; i++)
                {
                    uin32_t v_index = face_0 + i;
                    const float* w = weight_ptr + 6 * v_index;
                    float* out = out_ptr + v_index * G;

                    float value = in_ptr[v_index];
                    out[0] = w[0] * (in_ptr[wrap(v_index + base_off[0], V)] - value);
                    out[1] = w[1] * (in_ptr[wrap(v_index + base_off[1], V)] - value);
                    out[2] = w[2] * (in_ptr[wrap(v_index + base_off[2], V)] - value) +
                             w[3] * (in_ptr[wrap(v_index + base_off[3], V)] - value);
                    out[3] = w[4] * (in_ptr[wrap(v_index + base_off[4], V)] - value) + 
                             w[5] * (in_ptr[wrap(v_index + base_off[5], V)] - value);
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_weight_table", &create_weight_table, "Create a weight table for gradients")
    m.def("icosphere_gradient", &icosphere_gradient_cpu, "Calculate icosphere gradient");
    m.def("get_icosphere_level", &get_icosph_level, "Get the level of an icosphere");
}