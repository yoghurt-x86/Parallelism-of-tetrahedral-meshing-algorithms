#include "TetMesh.h"
#include <igl/readSTL.h>
#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#ifdef HIP_ENABLED
#include "../hip/vertex_processor.h"
#endif

int main(int argc, char *argv[]) {
    using namespace std;
    using namespace Eigen;
    
    //const std::string mesh_file = "../../libigl/meshes/53754.stl";
    //const std::string mesh_file = "../../libigl/meshes/tetrahedron.obj";
    const std::string mesh_file = "../../libigl/meshes/spot_triangulated.obj";
    //const std::string mesh_file = "../../libigl/meshes/cube.obj";

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXi N;
    
    //std::ifstream stl_file(mesh_file);
    //if(!igl::readSTL(stl_file, V, F, N)){
    //    std::cout << "Failed to load STL file: " << mesh_file << std::endl;
    //    return 1;
    //}
    //stl_file.close();

    if(!igl::readOBJ(mesh_file, V, F)){
        std::cout << "Failed to load OBJ file: " << mesh_file << std::endl;
        return 1;
    }

    std::cout << "Loaded mesh with " << V.rows() << " vertices and " << F.rows() << " faces" << std::endl;

    std::cout << "\nGenerating TetGen tetrahedralization..." << std::endl;
    MatrixXd TV;
    MatrixXi TT;
    MatrixXi TF;

    Eigen::VectorXi VM, FM;
    Eigen::MatrixXd H, Reg;
    Eigen::VectorXi TM, TR, PT;
    Eigen::MatrixXi FT, TN;
    int numRegions;

    //igl::copyleft::tetgen::tetrahedralize(V,F, H, VM, FM, Reg, "pq1.414a0.1n", TV,TT,TF,TM, TR, TN, PT, FT, numRegions );
    igl::copyleft::tetgen::tetrahedralize(V, F, H, VM, FM, Reg, "pq1.414n", TV, TT, TF, TM, TR, TN, PT, FT, numRegions);

#ifdef HIP_ENABLED
    std::cout << "GPU acceleration available" << std::endl;
    std::cout << "Scaling vertices with GPU..." << std::endl;
    VertexProcessor::processVerticesFromPointer(V.data(), V.rows(), 1.05f);
    VertexProcessor::printGPUInfo();
#else
    std::cout << "GPU acceleration not available" << std::endl;
#endif
    
    std::cout << "\nProcessing complete!" << std::endl;


    MatrixXi AM;
    VectorXi prefix_sum; 
    VectorXi indexes; 
    TetMesh::adjacency(TT,TV,AM);
    TetMesh::csr_from_AM(AM, prefix_sum, indexes);

    //std::cout << AM.size() << std::endl;
    //std::cout << "prfx sum " << prefix_sum.size() << " - rows " << TV.rows() << std::endl;
    //std::cout << "sum val " << prefix_sum(prefix_sum.size()-1) << std::endl;
    //std::cout << "hehe  "<< indexes.size() << std::endl;

    //MatrixXi edges;
    //TetMesh::edge_pairs_from_TT(TT, edges);
    //Eigen::MatrixXi flips23, flips32, TF23;
    //TetMesh::flips(TT, TN, flips23, flips32, TF23);

    //TetMesh::flip32(flips32(0, 0), flips32(0, 1), flips32(0, 2), TT, TN, TV);

    std::cout << "TT: \n" << TT << std::endl;
    std::cout << "TN: \n" << TN << std::endl;
    //std::cout << "TF23: \n" << TF23 << std::endl;
    //std::cout << "flips: \n" << flips23 << std::endl;

    MatrixXd TV_gpu = TV.transpose();
    MatrixXi TT_gpu = TT.transpose();
    MatrixXi TN_gpu = TN.transpose();

    MatrixXi TT_out_gpu(4, TT.rows()*3);
    MatrixXi TN_out_gpu(4, TN.rows()*3);
    //MatrixXi flips23_gpu = flips23.transpose();
    //MatrixXi flips32_gpu = flips32.transpose();
    //MatrixXi TF23_gpu = TF23.transpose();

    //MatrixXi edges_gpu = edges.transpose();
    //VectorXi prefix_sum_gpu = prefix_sum;

#ifdef HIP_ENABLED
    //VertexProcessor::smooth_tets_naive(TV_gpu.data(), TV.rows(), edges_gpu.data(), edges.rows(), prefix_sum_gpu.data());
    VertexProcessor::flip_23(TV_gpu.data(), TV.rows(),
                             TT_gpu.data(), TN_gpu.data(), TT.rows(),
                             TT_out_gpu.data(), TN_out_gpu.data()
                             );
#endif

    MatrixXi TT_out;
    MatrixXi TN_out;

    TT_out.noalias() = TT_out_gpu.transpose();
    TN_out.noalias() = TN_out_gpu.transpose();

    //flips23.noalias() = flips23_gpu.transpose();
    //TV.noalias() = TV_gpu.transpose();
    //edges.noalias() = edges_gpu.transpose();
    //prefix_sum.noalias() = prefix_sum_gpu;

    //std::cout << "V" << TV << std::endl;
    //std::cout << "edges:" << edges << std::endl;
    //std::cout << "sum:" << prefix_sum << std::endl;

    //std::cout << "flips: \n" << flips23 << std::endl;

    return 0;
}
