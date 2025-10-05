#pragma once

#include <vector>

namespace VertexProcessor {
    void processVertices(std::vector<float>& vertices, int vertex_count, float scale_factor = 1.1f);
    void processVerticesFromPointer(double* vertices, int vertex_count, float scale_factor = 1.1f);
    
    void translateVertices(std::vector<float>& vertices, int vertex_count, 
                          float dx = 0.0f, float dy = 0.0f, float dz = 0.0f);
    void translateVerticesFromPointer(double* vertices, int vertex_count, 
                                    float dx = 0.0f, float dy = 0.0f, float dz = 0.0f);
    void smooth_tets_naive(double* TV, int vertex_count, int* edge_pairs, int num_edges, int* prefix_sum);

    void printGPUInfo();
}
