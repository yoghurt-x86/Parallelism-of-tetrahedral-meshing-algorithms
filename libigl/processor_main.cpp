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
    igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Y", TV, TT, TF);

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
    std::cout << "prfx sum " << prefix_sum.size() << " - rows " << TV.rows() << std::endl;
    std::cout << "sum val " << prefix_sum(prefix_sum.size()-1) << std::endl;
    std::cout << "hehe  "<< indexes.size() << std::endl;

    MatrixXi edges; 
    TetMesh::edge_pairs_from_TT(TT, edges);

#ifdef HIP_ENABLED
    VertexProcessor::smooth_tets_naive(V.data(), V.rows(), edges.data(), edges.rows(), prefix_sum.data());
#endif

    std::cout << "edges" << edges.size() << std::endl;

    std::cout << "V" << TV << std::endl;
    std::cout << "edges:" << edges << std::endl;
    std::cout << "sum:" << prefix_sum << std::endl;

    return 0;
}
