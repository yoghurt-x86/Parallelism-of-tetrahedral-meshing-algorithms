#include "vertex_processor.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <hipcub/hipcub.hpp>
#include <Eigen/Dense>


__global__ void set_ones_kernel(int* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1;
    }
}

__global__ void scatter_kernel(int* input, int* indices,
                                int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[indices[tid]] = input[tid];
    }
}

__global__ void scatterTets(
    int* TT,
    int* TT_counts,
    int* TT_offsets,
    int* TT_out,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int count = TT_counts[tid];
    if (count != 1) return; // This tet no longer exists

    int out_idx = TT_offsets[tid];

    for (int i = 0; i<4; ++i) {
        TT_out[out_idx*4+i] = TT[tid*4+i];
    }
}


__global__ void scaleVerticesKernel(float* vertices, int vertex_count, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vertex_count * 3) {
        vertices[idx] *= scale_factor;
    }
}

__global__ void scaleVerticesKernelDouble(double* vertices, int vertex_count, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vertex_count * 3) {
        vertices[idx] *= scale_factor;
    }
}

__global__ void translateVerticesKernel(float* vertices, int vertex_count, float dx, float dy, float dz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vertex_idx = idx / 3;
    int coord_idx = idx % 3;
    
    if (vertex_idx < vertex_count) {
        if (coord_idx == 0) {
            vertices[idx] += dx;
        } else if (coord_idx == 1) {
            vertices[idx] += dy;
        } else if (coord_idx == 2) {
            vertices[idx] += dz;
        }
    }
}

__global__ void translateVerticesKernelDouble(double* vertices, int vertex_count, float dx, float dy, float dz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vertex_idx = idx / 3;
    int coord_idx = idx % 3;
    
    if (vertex_idx < vertex_count) {
        if (coord_idx == 0) {
            vertices[idx] += dx;
        } else if (coord_idx == 1) {
            vertices[idx] += dy;
        } else if (coord_idx == 2) {
            vertices[idx] += dz;
        }
    }
}

__global__ void smooth_by_edges(double* V, int* E, int *prefix_sum, int edge_count, double* V_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < edge_count) {
      int first  = E[(idx*2)];
      int second = E[(idx*2)+1];

      double sum1 = 1 / (double) (first == 0 ? prefix_sum[first] : prefix_sum[first] - prefix_sum[first-1]);
      double sum2 = 1 / (double) (second == 0 ? prefix_sum[second] : prefix_sum[second] - prefix_sum[second-1]);

      first  *= 3;
      second *= 3;

      atomicAdd(&V_out[first + 0], V[second + 0] * (sum1));
      atomicAdd(&V_out[first + 1], V[second + 1] * (sum1));
      atomicAdd(&V_out[first + 2], V[second + 2] * (sum1));

      atomicAdd(&V_out[second + 0], V[first + 0] * (sum2));
      atomicAdd(&V_out[second + 1], V[first + 1] * (sum2));
      atomicAdd(&V_out[second + 2], V[first + 2] * (sum2));
    }
}

//signed_volume(
//        TV.row(tet1_r(0)),
//        TV.row(tet1_r(2)),
//        TV.row(tet1_r(1)),
//        TV.row(tet1_r(3))
//    );

__device__ inline double min_dihedral_angle(double* TV, Eigen::Vector4i tet) {
    using namespace Eigen;
    using namespace std;

	Vector3d a(TV[tet(0)*3+0], TV[tet(0)*3+1], TV[tet(0)*3+2]);
	Vector3d b(TV[tet(2)*3+0], TV[tet(2)*3+1], TV[tet(2)*3+2]);
	Vector3d c(TV[tet(1)*3+0], TV[tet(1)*3+1], TV[tet(1)*3+2]);
	Vector3d d(TV[tet(3)*3+0], TV[tet(3)*3+1], TV[tet(3)*3+2]);

    Eigen::Matrix<double, 4, 3> normal(4, 3);
    normal.row(0) = (a - c).cross(d - c).normalized();
    normal.row(1) = (c - b).cross(d - b).normalized();
    normal.row(2) = (b - a).cross(d - a).normalized();
    normal.row(3) = (a - b).cross(c - b).normalized();

    Matrix<double, 6, 1> cartesian_normals;
    unsigned int k = 0;
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned m = n + 1; m < 4; ++m) {
            //auto cos_angle = clamp(normal.row(n).dot(normal.row(m)), -1.0, 1.0);
            auto cos_angle = normal.row(n).dot(normal.row(m));
            cartesian_normals(k) = M_PI - std::acos(cos_angle);
            k++;
        }
    }
    return cartesian_normals.minCoeff();
}

__device__ inline double signed_volume(double* TV, Eigen::Vector4i tet) {
    using namespace Eigen;

    Matrix3d det;
    det << (TV[tet(0)*3+0] - TV[tet(3)*3+0]), (TV[tet(2)*3+0] - TV[tet(3)*3+0]), (TV[tet(1)*3+0] - TV[tet(3)*3+0])
         , (TV[tet(0)*3+1] - TV[tet(3)*3+1]), (TV[tet(2)*3+1] - TV[tet(3)*3+1]), (TV[tet(1)*3+1] - TV[tet(3)*3+1])
         , (TV[tet(0)*3+2] - TV[tet(3)*3+2]), (TV[tet(2)*3+2] - TV[tet(3)*3+2]), (TV[tet(1)*3+2] - TV[tet(3)*3+2]);

    return det.determinant() * (1.0 / 6.0);
}

// Kernel 1: Identify candidate flips from topology
__global__ void identify_flips(
    int* TT,           // Tetrahedra (tet_count × 4)
    int* TN,           // Tet neighbors (tet_count × 4)
    int tet_count,
    int* candidate_faces,     // Output: face pairs (max_faces × 2)
	int* tet_to_flips, 		  // Output: Four flips per tet
//    int* face_to_tets,        // Output: which tets share this face (max_faces × 2)
//    int* face_to_local_idx,   // Output: local face index in each tet (max_faces × 2)
    unsigned int* candidate_count      // Output: atomic counter
) {
    int tet_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet_idx >= tet_count) return;

    // Check each of the 4 faces of this tetrahedron
    for (int i = 0; i < 4; ++i) {
        int neighbor_tet = TN[tet_idx * 4 + i];

        // Only process each face once (avoid duplicates)
        if (neighbor_tet < 0 || neighbor_tet <= tet_idx) continue;

        unsigned int slot = atomicAdd(candidate_count, 1);

        candidate_faces[slot * 2 + 0] = tet_idx;
        candidate_faces[slot * 2 + 1] = neighbor_tet;

        for (int i = 0; i < 4; ++i) {
            int old = atomicCAS(&tet_to_flips[tet_idx * 4 + i], -1, slot);
            if (old == -1) break;
        }

        for (int i = 0; i < 4; ++i) {
            int old = atomicCAS(&tet_to_flips[neighbor_tet * 4 + i], -1, slot);
            if (old == -1) break;
        }
    }
}

__global__ void flip_faces(double* TV,      int  vertex_count,
             int*    TT,      int* TN,     	int* TF23, 		int tet_count,
             int*    flips23, int  flip_count,
             double* flip_quality
) {
	using namespace Eigen;
	using namespace std;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= flip_count) return;

    int flipidx1 = flips23[idx*2+0];
    int flipidx2 = flips23[idx*2+1];

    Vector4i tet1(TT[flipidx1*4+0],TT[flipidx1*4+1],TT[flipidx1*4+2],TT[flipidx1*4+3]);
    Vector4i tet2(TT[flipidx2*4+0],TT[flipidx2*4+1],TT[flipidx2*4+2],TT[flipidx2*4+3]);

    int shared[3];
	{
        unsigned c = 0;
        for (unsigned i = 0; i < 4; ++i) {
            for (unsigned j = 0; j < 4; ++j) {
                int p1 = tet1(i);
                int p2 = tet2(j);
                if (p1 == p2) {
                    shared[c] = p1;
                    c++;
                }
            }
        }

        assert(c == 3);
    }

    // Find the apex vertices (the vertices not on the shared face)
    int apex1 = -1, apex2 = -1;
    for (int v = 0; v < 4; ++v) {
        bool flag = false;
        for (int i = 0; i < 3; ++i) {
            if (shared[i] == tet1(v))
                flag = true;
        }
        if (!flag) {
            apex1 = tet1(v);
            break;
        }
    }

    for (int v = 0; v < 4; ++v) {
        bool flag = false;
        for (int i = 0; i < 3; ++i) {
            if (shared[i] == tet2(v))
                flag = true;
        }
        if (!flag) {
            apex2 = tet2(v);
            break;
        }
    }

    assert(apex1 != -1 && apex2 != -1);

    Vector3d a = Vector3d(TV[shared[0]*3+0],TV[shared[0]*3+1],TV[shared[0]*3+2]);
    Vector3d b = Vector3d(TV[shared[1]*3+0],TV[shared[1]*3+1],TV[shared[1]*3+2]);
    Vector3d c = Vector3d(TV[shared[2]*3+0],TV[shared[2]*3+1],TV[shared[2]*3+2]);
    Vector3d p = Vector3d(TV[apex1*3+0],TV[apex1*3+1],TV[apex1*3+2]);

    Vector3d ab = b - a;
    Vector3d ac = c - a;
    Vector3d ap = p - a;

    auto orientation_test = ap.dot(ab.cross(ac));
    if (orientation_test < 0.0){
        //swap(shared[1], shared[2]);
		int tmp = shared[1];
		shared[1] = shared[2];
		shared[2] = tmp;
	}

    Vector4i tet1_r(apex2, apex1, shared[0], shared[1]);
    Vector4i tet2_r(apex2, apex1, shared[1], shared[2]);
    Vector4i tet3_r(apex2, apex1, shared[2], shared[0]);

    double V_1 = signed_volume(TV, tet1_r);
    double V_2 = signed_volume(TV, tet2_r);
    double V_3 = signed_volume(TV, tet3_r);


    if (V_1 <= 0.0 || V_2 <= 0.0 || V_3 <= 0.0) {
		flip_quality[idx] = 0.0;
    } else {
        double T1_result = min_dihedral_angle(TV, tet1_r);
        double T2_result = min_dihedral_angle(TV, tet2_r);
        double T3_result = min_dihedral_angle(TV, tet3_r);

		double T1_before = min_dihedral_angle(TV, tet1);
		double T2_before = min_dihedral_angle(TV, tet2);

		double old_min = std::min(T1_result, T2_result);
		double new_min = std::min(min(T1_result, T2_result), T3_result);

		double improvement = new_min - old_min;
		flip_quality[idx] = improvement;

		// For debug purpose:
		if (improvement == 0.0) {
            flip_quality[idx] = 1.0;
        }

		//// Higher is better
		//if (old_min >= new_min) {
		//	// No improvement
        //    //flip_out_c[idx*2] = 0;
		//	// flip anyway
        //    flip_quality[idx] = -1.0;
        //    //flip_out_c[idx*2] = 0;
        //} else {
        //    flip_out_c[idx*2] = 1;
		//}

		// todo: fix this. this has to be device wide sync
		//__syncthreads();

		//bool to_flip = true;
		//for (int i = 0; i < 4; ++i) {
		//	int flip = TF23[flipidx1*4+i];
		//	if(flip == idx)
		//		continue;
		//	if(flip_quality[flip] > improvement) {
		//		to_flip = false;
		//		break;
		//	}
		//}
		//for (int i = 0; i < 4; ++i) {
		//	int flip = TF23[flipidx2*4+i];
		//	if(flip == idx)
		//		continue;
		//	if(flip_quality[flip] > improvement) {
		//		to_flip = false;
		//		break;
		//	}
		//}
        //flip_out_c[idx*2+1] = to_flip? 1 : 0;
		//result = 1.0;
	}
}

__device__ inline int matching_neighbor_index(int flipIndex1, int flipIndex2, int* TT, int* TN, Eigen::Vector4i tet, Eigen::Vector4i neighbors) {
    using namespace Eigen;

    for (int i = 0; i<4; ++i) {
        if(neighbors(i) == -1 || neighbors(i) == flipIndex1 || neighbors(i) == flipIndex2) continue;
        Vector4i ntet(TT[neighbors[i]*4+0], TT[neighbors[i]*4+1],TT[neighbors[i]*4+2],TT[neighbors[i]*4+3]);
        int count = 0;
        for(int j=0; j<4; ++j){
            for(int k=0; k<4; ++k){
                if(ntet(j) == tet(k)) count++;
            }
        }
        if (count == 3) {
            return neighbors(i);
        }
    }
    return -1;
}


__global__ void write_flipped_faces(
             double* TV,      int  vertex_count,
             int*    TT,      int* TN,     	int* TT_offset,   int tet_count,
             int*    flips23, int  flip_count,
             int*    flip_offset, int* flip_counts,
             int*    TT_out,  int*    TN_out
) {
	using namespace Eigen;
	using namespace std;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= flip_count) return;
    if (flip_counts[idx] == 0) return;

    int flipidx1 = flips23[idx*2+0];
    int flipidx2 = flips23[idx*2+1];

    Vector4i tet1(TT[flipidx1*4+0],TT[flipidx1*4+1],TT[flipidx1*4+2],TT[flipidx1*4+3]);
    Vector4i tet2(TT[flipidx2*4+0],TT[flipidx2*4+1],TT[flipidx2*4+2],TT[flipidx2*4+3]);

    int shared[3];
	{
        unsigned c = 0;
        for (unsigned i = 0; i < 4; ++i) {
            for (unsigned j = 0; j < 4; ++j) {
                int p1 = tet1(i);
                int p2 = tet2(j);
                if (p1 == p2) {
                    shared[c] = p1;
                    c++;
                }
            }
        }

        assert(c == 3);
    }

    // Find the apex vertices (the vertices not on the shared face)
    int apex1 = -1, apex2 = -1;
    for (int v = 0; v < 4; ++v) {
        bool flag = false;
        for (int i = 0; i < 3; ++i) {
            if (shared[i] == tet1(v))
                flag = true;
        }
        if (!flag) {
            apex1 = tet1(v);
            break;
        }
    }

    for (int v = 0; v < 4; ++v) {
        bool flag = false;
        for (int i = 0; i < 3; ++i) {
            if (shared[i] == tet2(v))
                flag = true;
        }
        if (!flag) {
            apex2 = tet2(v);
            break;
        }
    }

    assert(apex1 != -1 && apex2 != -1);

    Vector3d a = Vector3d(TV[shared[0]*3+0],TV[shared[0]*3+1],TV[shared[0]*3+2]);
    Vector3d b = Vector3d(TV[shared[1]*3+0],TV[shared[1]*3+1],TV[shared[1]*3+2]);
    Vector3d c = Vector3d(TV[shared[2]*3+0],TV[shared[2]*3+1],TV[shared[2]*3+2]);
    Vector3d p = Vector3d(TV[apex1*3+0],TV[apex1*3+1],TV[apex1*3+2]);

    Vector3d ab = b - a;
    Vector3d ac = c - a;
    Vector3d ap = p - a;

    auto orientation_test = ap.dot(ab.cross(ac));
    if (orientation_test < 0.0){
        //swap(shared[1], shared[2]);
		int tmp = shared[1];
		shared[1] = shared[2];
		shared[2] = tmp;
	}

    Vector4i tet1_r(apex2, apex1, shared[0], shared[1]);
    Vector4i tet2_r(apex2, apex1, shared[1], shared[2]);
    Vector4i tet3_r(apex2, apex1, shared[2], shared[0]);

    Vector4i tet_neighbors1(TN[flipidx1*4+0],TN[flipidx1*4+1],TN[flipidx1*4+2],TN[flipidx1*4+3]);
    Vector4i tet_neighbors2(TN[flipidx2*4+0],TN[flipidx2*4+1],TN[flipidx2*4+2],TN[flipidx2*4+3]);


    int tet_1_n1 = matching_neighbor_index(flipidx1, flipidx2, TT, TN, tet1_r, tet_neighbors1);
    int tet_1_n2 = matching_neighbor_index(flipidx1, flipidx2, TT, TN, tet1_r, tet_neighbors2);

    int tet_2_n1 = matching_neighbor_index(flipidx1, flipidx2, TT, TN, tet2_r, tet_neighbors1);
    int tet_2_n2 = matching_neighbor_index(flipidx1, flipidx2, TT, TN, tet2_r, tet_neighbors2);

    int tet_3_n1 = matching_neighbor_index(flipidx1, flipidx2, TT, TN, tet3_r, tet_neighbors1);
    int tet_3_n2 = matching_neighbor_index(flipidx1, flipidx2, TT, TN, tet3_r, tet_neighbors2);

    int idx_out = TT_offset[flipidx1];

    TN_out[idx_out*4+0+0] = tet_1_n1;
    TN_out[idx_out*4+0+1] = tet_1_n2;
    TN_out[idx_out*4+0+2] = -2;
    TN_out[idx_out*4+0+3] = -2;

    TN_out[idx_out*4+4+0] = tet_2_n1;
    TN_out[idx_out*4+4+1] = tet_2_n2;
    TN_out[idx_out*4+4+2] = -2;
    TN_out[idx_out*4+4+3] = -2;

    TN_out[idx_out*4+8+0] = tet_3_n1;
    TN_out[idx_out*4+8+1] = tet_3_n2;
    TN_out[idx_out*4+8+2] = -2;
    TN_out[idx_out*4+8+3] = -2;

    for (int i = 0; i<4; ++i){
        TT_out[idx_out*4+0+i] = tet1_r(i);
        TT_out[idx_out*4+4+i] = tet2_r(i);
        TT_out[idx_out*4+8+i] = tet3_r(i);
    }
}

__global__ void apply_flips(double* TV,      int  vertex_count,
             int*    TT,      int* TN,     	int* TF23, 		int tet_count,
             int*    TT_sum,
             int*    flips23, int  flip_count,
             double* flip_quality, int* flips23_count
) {
	using namespace Eigen;
	using namespace std;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= flip_count) return;
	if (flip_quality[idx] <= 0.0) return;

    int flipidx1 = flips23[idx*2+0];
    int flipidx2 = flips23[idx*2+1];

	double improvement = flip_quality[idx];
    bool to_flip = true;
    for (int i = 0; i < 4; ++i) {
        int flip = TF23[flipidx1*4+i];
        if(flip == idx)
            continue;
        if(flip_quality[flip] > improvement) {
            to_flip = false;
            break;
        }
    }
    for (int i = 0; i < 4; ++i) {
        int flip = TF23[flipidx2*4+i];
        if(flip == idx)
            continue;
        if(flip_quality[flip] > improvement) {
            to_flip = false;
            break;
        }
    }

	if(to_flip) {
		TT_sum[flipidx1] = 3;
		TT_sum[flipidx2] = 0;
        flips23_count[idx] = 3;
	} else {
		// This is ok cause we know this is not the highest value in this neighborhood and will not flip hehe!
    	flip_quality[idx] = 0.0;
        flips23_count[idx] = 0;
	}
}

__device__ inline bool is_neighbors(Eigen::Vector4i tet1, Eigen::Vector4i tet2) {
    using namespace Eigen;

    int count;
    for(int j=0; j<4; ++j){
        for(int k=0; k<4; ++k){
            if(tet1(j) == tet2(k)) count++;
        }
    }

    assert(count <= 3);
    return count == 3;
}

__global__ void update_neighbors(
             double* TV,      int  vertex_count,
             int*    TT,      int* TN,     	int* TF23, 		int tet_count,
             int* d_TT_count, int* d_TT_sum,
             int*    flips23, int  flip_count
) {
    using namespace Eigen;
    using namespace std;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tet_count) return;

    for (int i  = 0; i<4; ++i) {
        int neighbor = TN[idx*4+i];

        if (neighbor == -1) continue;
        if (neighbor == -2) continue;

        int num_neigh_candidates = d_TT_count[neighbor];
        if (num_neigh_candidates == 1) {
            TN[idx*4+i] = d_TT_sum[neighbor];
            continue;
        }

        int flipidx = TF23[neighbor];
        int tet1 = flips23[flipidx*2+0];

        Vector4i tet(TT[idx*4+0],TT[idx*4+1],TT[idx*4+2],TT[idx*4+3]);
        for (int j = 0; j<d_TT_count[neighbor]; ++j) {
            Vector4i neighbor_candidate(TT[tet1*4+(j*4)+0],TT[tet1*4+(j*4)+1],TT[tet1*4+(j*4)+2],TT[tet1*4+(j*4)+3]);
            if (is_neighbors(tet, neighbor_candidate)) {
                TN[idx*4+i] = tet1+j;
                break;
            }
        }
    }
}

__global__ void finish_neighbors(
             double* TV,      int  vertex_count,
             int*    TT,      int* TN,     	int* TF23, 		int tet_count,
             int* d_TT_count, int* d_TT_sum,
             int*    flips23, int* flip_counts, int flip_count
) {
    using namespace Eigen;
    using namespace std;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= flip_count) return;

    if (flip_counts[idx] != 0) {
        int flipidx = flips23[idx*2+0];

        int tet1 = d_TT_sum[flipidx];

        TN[tet1*4+0+2] = tet1+1;
        TN[tet1*4+0+3] = tet1+2;

        TN[tet1*4+4+2] = tet1+0;
        TN[tet1*4+4+3] = tet1+2;

        TN[tet1*4+8+2] = tet1+0;
        TN[tet1*4+8+3] = tet1+1;
    }
}

namespace VertexProcessor {

void processVertices(std::vector<float>& vertices, int vertex_count, float scale_factor) {
    float* d_vertices;
    size_t size = vertex_count * 3 * sizeof(float);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices.data(), size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(scaleVerticesKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, scale_factor);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices.data(), d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Scaled " << vertex_count << " vertices by factor " << scale_factor << std::endl;
}

void translateVertices(std::vector<float>& vertices, int vertex_count, float dx, float dy, float dz) {
    float* d_vertices;
    size_t size = vertex_count * 3 * sizeof(float);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices.data(), size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(translateVerticesKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, dx, dy, dz);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices.data(), d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Translated " << vertex_count << " vertices by (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
}

void processVerticesFromPointer(double* vertices, int vertex_count, float scale_factor) {
    double* d_vertices;
    size_t size = vertex_count * 3 * sizeof(double);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices, size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(scaleVerticesKernelDouble, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, scale_factor);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices, d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Scaled " << vertex_count << " vertices by factor " << scale_factor << std::endl;
}

void translateVerticesFromPointer(double* vertices, int vertex_count, float dx, float dy, float dz) {
    double* d_vertices;
    size_t size = vertex_count * 3 * sizeof(double);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices, size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(translateVerticesKernelDouble, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, dx, dy, dz);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices, d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Translated " << vertex_count << " vertices by (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
}

void flip_23(double* TV,      int  vertex_count,
             int*    TT,      int* TN,     		int tet_count
             ) {
    double* d_TV;
    int* d_TT_in;
    int* d_TN_in;
    //int* d_TF23_in;
    //int* d_flips23_in;

    double* d_flip_quality;
    int* d_flip_counts;
	double* flip_quality;

    size_t size_verts = vertex_count * 3 * sizeof(double);
    size_t size_tets = tet_count * 4 * sizeof(int);

	// Init inputs:
    hipMalloc(&d_TV, size_verts);
    hipMalloc(&d_TT_in, size_tets);
    hipMalloc(&d_TN_in, size_tets);
    //hipMalloc(&d_TF23_in, size_tets);
    //hipMalloc(&d_flips23_in, size_flips);

    hipMemcpy(d_TV, TV, size_verts, hipMemcpyHostToDevice);
    hipMemcpy(d_TT_in, TT, size_tets, hipMemcpyHostToDevice);
    hipMemcpy(d_TN_in, TN, size_tets, hipMemcpyHostToDevice);
    //hipMemcpy(d_TF23_in, TN, size_tets, hipMemcpyHostToDevice);
    //hipMemcpy(d_flips23_in, flips23 , size_flips, hipMemcpyHostToDevice);

	// Init outputs:


	// Candidate flips
    //std::cout << "=== Count flips: " << std::endl;
	unsigned int* d_candidate_count;
    int* d_flips23_candidates;
    int* d_TF23_out;


	unsigned int candidate_count;

    hipMalloc(&d_candidate_count, sizeof(unsigned int));
    hipMemset(d_candidate_count, 0, sizeof(unsigned int));

	size_t max_flips = tet_count * 4 / 2;
	size_t max_flips_size = max_flips * 2 * sizeof(int);
    hipMalloc(&d_flips23_candidates, max_flips_size);

    hipMalloc(&d_TF23_out, size_tets);
    hipMemset(d_TF23_out, -1, size_tets);


    int threadsPerBlockI = 256;
    int blocksPerGridEdgesI = (tet_count + threadsPerBlockI - 1) / threadsPerBlockI;

    hipLaunchKernelGGL(identify_flips, dim3(blocksPerGridEdgesI), dim3(threadsPerBlockI), 0, 0,
		d_TT_in, d_TN_in, tet_count,
		d_flips23_candidates,  d_TF23_out, d_candidate_count);
    hipDeviceSynchronize();
    hipMemcpy(&candidate_count, d_candidate_count, sizeof(unsigned int), hipMemcpyDeviceToHost);

    size_t size_flips = candidate_count * 2 * sizeof(int);
    size_t size_flip_counts = candidate_count * sizeof(int);
    size_t size_flip_quality = candidate_count * sizeof(double);
    hipMalloc(&d_flip_quality, size_flip_quality);
    flip_quality = (double *) std::malloc(size_flip_quality);
    hipMalloc(&d_flip_counts, size_flip_counts);
    hipMemset(d_flip_counts, 0, size_flip_counts);
    //std::cout << "Number of flips: " << candidate_count << " out of: " << max_flips << std::endl;
	//{
    //    int* flips23_candidates;
    //    int* TF23_out;
    //    flips23_candidates = (int *) malloc(max_flips_size);
    //    TF23_out = (int *) malloc(size_tets);

    //    hipMemcpy(flips23_candidates, d_flips23_candidates, max_flips_size, hipMemcpyDeviceToHost);
    //    hipMemcpy(TF23_out, d_TF23_out, size_tets, hipMemcpyDeviceToHost);

    //    std::cout << "flips!!! gpu: " << std::endl;
    //    for (int i = 0; i < candidate_count; ++i) {
    //        std::cout << flips23_candidates[i*2+0] << " " << flips23_candidates[i*2+1] << std::endl;
    //    }

    //    std::cout << "tf23 !!! gpu: " << std::endl;
    //    for (int i = 0; i < tet_count; ++i) {
    //        for (int j = 0; j < 4; ++j) {
    //            std::cout << TF23_out[i*4+j] << " ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << "Stop Count flips." << std::endl;
    //}

	// end candidate flips

    int threadsPerBlock = 256;
    int blocksPerGridEdges = (candidate_count + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(flip_faces, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0,
		d_TV, vertex_count,
		d_TT_in, d_TN_in, d_TF23_out, tet_count,
		d_flips23_candidates, candidate_count,
		d_flip_quality);
    hipDeviceSynchronize();

    int* d_TT_count;
	size_t size_tet_count = tet_count * sizeof(int);
    hipMalloc(&d_TT_count, size_tet_count);

    // Launch it
    int block_size = 256;
    int grid_size = (tet_count + block_size - 1) / block_size;
    hipLaunchKernelGGL(set_ones_kernel, grid_size, block_size, 0, 0,
                       d_TT_count, tet_count);

    hipDeviceSynchronize();

    hipLaunchKernelGGL(apply_flips, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0,
		d_TV, vertex_count,
		d_TT_in, d_TN_in, d_TF23_out, tet_count,
		d_TT_count,
		d_flips23_candidates, candidate_count,
		d_flip_quality, d_flip_counts);
    hipDeviceSynchronize();

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int* d_TT_sum;
    hipMalloc(&d_TT_sum, size_tet_count);
	hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                     d_TT_count, d_TT_sum, tet_count);
    // Allocate temporary storage
    hipMalloc(&d_temp_storage, temp_storage_bytes);
	hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                     d_TT_count, d_TT_sum, tet_count);

    void* d_temp_storage2 = nullptr;
    size_t temp_storage_bytes2 = 0;

    int* d_flip_counts_sum;
    hipMalloc(&d_flip_counts_sum, size_flip_counts);

    hipcub::DeviceScan::ExclusiveSum(d_temp_storage2, temp_storage_bytes2,
                                     d_flip_counts, d_flip_counts_sum, candidate_count);
    // Allocate temporary storage
    hipMalloc(&d_temp_storage2, temp_storage_bytes2);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage2, temp_storage_bytes2,
                                     d_flip_counts, d_flip_counts_sum, candidate_count);

    hipDeviceSynchronize();

    int new_tet_sum;
    int new_tet_count;
    hipMemcpy(&new_tet_sum, &d_TT_sum[tet_count - 1], sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&new_tet_count, &d_TT_count[tet_count - 1], sizeof(int), hipMemcpyDeviceToHost);

    new_tet_count += new_tet_sum;

    std::cout <<  "Number of tets:  " << tet_count << std::endl;
    std::cout <<  "New number of tets:   " << new_tet_count << std::endl;

    //hipMemcpy(flip_quality, d_flip_quality, size_flip_quality, hipMemcpyDeviceToHost);

    //std::cout << "Qualities: " << std::endl;
	//for (int i = 0; i < flip_count; ++i) {
	//	std::cout << flip_quality[i] << std::endl;
	//}

    //int* TT_sum;
    //TT_sum = (int *) malloc(size_tet_count);
    //hipMemcpy(TT_sum, d_TT_count, size_tet_count, hipMemcpyDeviceToHost);

    //std::cout << "TT_ cuiunt: " << std::endl;
	//for (int i = 0; i < tet_count; ++i) {
	//	std::cout << TT_sum[i] << std::endl;
	//}

    //hipMemcpy(TT_sum, d_TT_sum, size_tet_count, hipMemcpyDeviceToHost);
    //std::cout << "TT sum: " << std::endl;
	//for (int i = 0; i < tet_count; ++i) {
	//	std::cout << TT_sum[i] << std::endl;
	//}

    //int* flip_sum;
    //flip_sum = (int *) malloc(size_flip_counts);
    //hipMemcpy(flip_sum, d_flip_counts, size_flip_counts, hipMemcpyDeviceToHost);

    //std::cout << "flip cuiunt: " << std::endl;
    //for (int i = 0; i < flip_count; ++i) {
    //    std::cout << flip_sum[i] << std::endl;
    //}

    //hipMemcpy(flip_sum, d_flip_counts_sum, size_flip_counts, hipMemcpyDeviceToHost);
    //std::cout << "flip sum: " << std::endl;
    //for (int i = 0; i < flip_count; ++i) {
    //    std::cout << flip_sum[i] << std::endl;
    //}


    // Create new TT and TN
    int* d_TT_out;
    int* d_TN_out;
    size_t size_tets_out = new_tet_count * 4 * sizeof(int);
    hipMalloc(&d_TT_out, size_tets_out);
    hipMalloc(&d_TN_out, size_tets_out);

    int threadsPerBlockOut = 256;
    int blocksPerGridEdgesOut = (new_tet_count + threadsPerBlockOut - 1) / threadsPerBlockOut;
    hipLaunchKernelGGL(scatterTets, dim3(blocksPerGridEdgesOut), dim3(threadsPerBlockOut), 0, 0,
        d_TT_in, d_TT_count, d_TT_sum, d_TT_out, tet_count);
    hipLaunchKernelGGL(scatterTets, dim3(blocksPerGridEdgesOut), dim3(threadsPerBlockOut), 0, 0,
        d_TN_in, d_TT_count, d_TT_sum, d_TN_out, tet_count);

    hipDeviceSynchronize();

    // Write new tets
    threadsPerBlock = 256;
    blocksPerGridEdges = (candidate_count + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(write_flipped_faces, dim3(blocksPerGridEdgesOut), dim3(threadsPerBlockOut), 0, 0,
        d_TV, vertex_count,
        d_TT_in, d_TN_in, d_TT_sum, tet_count,
        d_flips23_candidates, candidate_count,
        d_flip_counts_sum, d_flip_counts,
        d_TT_out, d_TN_out);

    hipDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGridEdges = (new_tet_count + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(update_neighbors, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0,
        d_TV, vertex_count,
        d_TT_out, d_TN_out, d_TF23_out, new_tet_count,
        d_TT_count, d_TT_sum,
        d_flips23_candidates, candidate_count);
    hipDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGridEdges = (candidate_count + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(finish_neighbors, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0,
        d_TV, vertex_count,
        d_TT_out, d_TN_out, d_TF23_out, new_tet_count,
        d_TT_count, d_TT_sum,
        d_flips23_candidates, d_flip_counts, candidate_count);
    hipDeviceSynchronize();

    //{ // DEBUG
    //    int* TT_out;
    //    TT_out = (int*) malloc(size_tets_out);
    //    hipMemcpy(TT_out, d_TT_out, size_tets_out, hipMemcpyDeviceToHost);

    //    std::cout << "TT !!! gpu out: " << std::endl;
    //    for (int i = 0; i < new_tet_count; ++i) {
    //        for (int j = 0; j < 4; ++j) {
    //            std::cout << TT_out[i*4+j] << " ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    hipMemcpy(TT_out, d_TN_out, size_tets_out, hipMemcpyDeviceToHost);

    //    std::cout << "TN !!! gpu out: " << std::endl;
    //    for (int i = 0; i < new_tet_count; ++i) {
    //        for (int j = 0; j < 4; ++j) {
    //            std::cout << TT_out[i*4+j] << " ";
    //        }
    //        std::cout << std::endl;
    //    }
    //}

    hipFree(d_flips23_candidates);
    hipFree(d_candidate_count);
    hipFree(d_TV);
    hipFree(d_TT_in);
    hipFree(d_TN_in);
    hipFree(d_flip_quality);
	free(flip_quality);
}

void smooth_tets_naive(double* TV, int vertex_count, int* edge_pairs, int num_edges, int* prefix_sum) {
    double* d_V;
    double* d_V_out;
    int* d_E;
    int* d_prefix_sum;

    size_t size_verts = vertex_count * 3 * sizeof(double);
    size_t size_edges = num_edges * 2 * sizeof(int);
    size_t size_prefix_sum = vertex_count * sizeof(int);
    
    hipMalloc(&d_V, size_verts);
    hipMalloc(&d_V_out, size_verts);
    hipMalloc(&d_E, size_edges);
    hipMalloc(&d_prefix_sum, size_prefix_sum);

    hipMemcpy(d_V, TV, size_verts, hipMemcpyHostToDevice);
    hipMemcpy(d_E, edge_pairs, size_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_prefix_sum, prefix_sum, size_prefix_sum, hipMemcpyHostToDevice);
    hipMemset(d_V_out, 0, size_verts);

    int threadsPerBlock = 256;
    int blocksPerGridEdges = (num_edges + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(smooth_by_edges, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0, d_V, d_E, d_prefix_sum, num_edges, d_V_out);
    hipDeviceSynchronize();

    hipMemcpy(TV, d_V_out, size_verts, hipMemcpyDeviceToHost);


    hipFree(d_V);
    hipFree(d_V_out);
    hipFree(d_E);
    hipFree(d_prefix_sum);
}

void printGPUInfo() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;
    
    for (int device = 0; device < deviceCount; device++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device);
        
        std::cout << "\nDevice " << device << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions: (" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Memory pitch: " << prop.memPitch / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Texture alignment: " << prop.textureAlignment << " bytes" << std::endl;
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Kernel execution timeout: " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Integrated GPU: " << (prop.integrated ? "Yes" : "No") << std::endl;
        std::cout << "  Can map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        
        // Get current memory usage
        size_t free_mem, total_mem;
        hipMemGetInfo(&free_mem, &total_mem);
        std::cout << "  Memory usage: " << (total_mem - free_mem) / (1024 * 1024) 
                  << " MB used / " << total_mem / (1024 * 1024) << " MB total" << std::endl;
    }
    
    // Example kernel launch configuration info
    std::cout << "\n=== Example Kernel Launch Configuration ===" << std::endl;
    int example_data_size = 10000;
    int threadsPerBlock = 256;
    int blocksPerGrid = (example_data_size + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "For " << example_data_size << " elements:" << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  Blocks per grid: " << blocksPerGrid << std::endl;
    std::cout << "  Total threads: " << blocksPerGrid * threadsPerBlock << std::endl;
    std::cout << "  Thread utilization: " << (float)example_data_size / (blocksPerGrid * threadsPerBlock) * 100 << "%" << std::endl;
}

}
