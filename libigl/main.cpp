#include "TetMesh.h"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/jet.h>
#include <igl/sort.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/make_conforming_constrained_Delaunay_triangulation_3.h>
#include <igl/barycenter.h>
#include <igl/stb/read_image.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#ifdef HIP_ENABLED
#include "../hip/vertex_processor.h"
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;

// Input mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;

Eigen::MatrixXd dV;
Eigen::MatrixXi dF;
Eigen::MatrixXd dC;
Eigen::VectorXd dHistogram;

TetMesh tetrahedra;
TetMesh delaunay;
TetMesh constrained_delaunay;

// matcap texture
Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;

enum Display {
  DISPLAY_INPUT,
  DISPLAY_TETGEN,
  DISPLAY_DELAUNAY,
  DISPLAY_CONSTRAINED_DELAUNAY,
};

enum ColorMap {
  COLORMAP_volumes,
  COLORMAP_av_ratio,
  COLORMAP_in_circum_ratio,
  COLORMAP_aspect_ratio,
  COLORMAP_dihedral_angles,
  COLORMAP_neighbors,
  COLORMAP_delaunay,
};

Display display = DISPLAY_INPUT;
ColorMap colorMap = COLORMAP_av_ratio;
double slice_t = 0.5;
double filter_t = 0.0;

void create_delaunay(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
  using namespace Eigen;
  // Create 3D Delaunay triangulation using CGAL
  Delaunay dt;
  std::vector<Point> points;

  MatrixXd DTV;
  MatrixXi DTT;
  MatrixXi DTF;

  // Convert Eigen vertices to CGAL points
  for(int i = 0; i < V.rows(); i++) {
    points.push_back(Point(V(i,0), V(i,1), V(i,2)));
  }

  // Insert points into Delaunay triangulation
  dt.insert(points.begin(), points.end());

  // Extract tetrahedra

  // Vertex extraction (same as above)
    std::map<Vertex_handle, int> vertex_index_map;
    std::vector<Point> vertices;

    int vertex_count = 0;
    for (auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
        vertex_index_map[vit] = vertex_count++;
        vertices.push_back(vit->point());
    }

    DTV.resize(vertex_count, 3);
    for (int i = 0; i < vertex_count; ++i) {
        DTV(i, 0) = CGAL::to_double(vertices[i].x());
        DTV(i, 1) = CGAL::to_double(vertices[i].y());
        DTV(i, 2) = CGAL::to_double(vertices[i].z());
    }

    // Tetrahedra extraction (same as above)
    std::vector<std::array<int, 4>> tetrahedra;
    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
        std::array<int, 4> tet;
        for (int i = 0; i < 4; ++i) {
            tet[i] = vertex_index_map[cit->vertex(i)];
        }
        tetrahedra.push_back(tet);
    }

    DTT.resize(tetrahedra.size(), 4);
    for (int i = 0; i < tetrahedra.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            DTT(i, j) = tetrahedra[i][j];
        }
    }

    // Extract ALL faces from tetrahedra
    std::set<std::array<int, 3>> unique_faces;

    for (const auto& tet : tetrahedra) {
        // Each tetrahedron has 4 faces
        std::array<std::array<int, 3>, 4> faces = {{
            {tet[0], tet[1], tet[2]},  // opposite to vertex 3
            {tet[0], tet[1], tet[3]},  // opposite to vertex 2
            {tet[0], tet[2], tet[3]},  // opposite to vertex 1
            {tet[1], tet[2], tet[3]}   // opposite to vertex 0
        }};

        for (auto& face : faces) {
            // Sort vertices to ensure consistent representation
            std::sort(face.begin(), face.end());
            unique_faces.insert(face);
        }
    }

    // Convert to matrix
    DTF.resize(unique_faces.size(), 3);
    int face_idx = 0;
    for (const auto& face : unique_faces) {
        for (int j = 0; j < 3; ++j) {
            DTF(face_idx, j) = face[j];
        }
        face_idx++;
    }


    delaunay = TetMesh(DTV, DTT, DTF);
}

void create_constrained_delaunay(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    using namespace Eigen;
    // Create surface mesh from the input mesh V,F
    SurfaceMesh mesh;
    MatrixXd CDTV;
    MatrixXi CDTT;
    MatrixXi CDTF;
    
    // Add vertices to the surface mesh
    std::vector<SurfaceMesh::Vertex_index> vertex_indices;
    for(int i = 0; i < V.rows(); i++) {
        auto vi = mesh.add_vertex(Point(V(i,0), V(i,1), V(i,2)));
        vertex_indices.push_back(vi);
    }
    
    // Add faces to the surface mesh
    for(int i = 0; i < F.rows(); i++) {
        std::vector<SurfaceMesh::Vertex_index> face_vertices = {
            vertex_indices[F(i,0)], 
            vertex_indices[F(i,1)], 
            vertex_indices[F(i,2)]
        };
        mesh.add_face(face_vertices);
    }
    
    // Create constrained Delaunay triangulation
    auto cdt = CGAL::make_conforming_constrained_Delaunay_triangulation_3(mesh);
    
    // Get the underlying triangulation
    const auto& triangulation = cdt.triangulation();
    
    // Extract tetrahedra from constrained Delaunay
    using Triangulation_type = typename std::remove_const<typename std::remove_reference<decltype(triangulation)>::type>::type;
    std::map<typename Triangulation_type::Vertex_handle, int> vertex_index_map;
    std::vector<Point> vertices;
    
    int vertex_count = 0;
    for (auto vit = triangulation.finite_vertices_begin(); vit != triangulation.finite_vertices_end(); ++vit) {
        vertex_index_map[vit] = vertex_count++;
        vertices.push_back(vit->point());
    }
    
    CDTV.resize(vertex_count, 3);
    for (int i = 0; i < vertex_count; ++i) {
        CDTV(i, 0) = CGAL::to_double(vertices[i].x());
        CDTV(i, 1) = CGAL::to_double(vertices[i].y());
        CDTV(i, 2) = CGAL::to_double(vertices[i].z());
    }
    
    // Extract tetrahedra
    std::vector<std::array<int, 4>> tetrahedra;
    for (auto cit = triangulation.finite_cells_begin(); cit != triangulation.finite_cells_end(); ++cit) {
        std::array<int, 4> tet;
        for (int i = 0; i < 4; ++i) {
            tet[i] = vertex_index_map[cit->vertex(i)];
        }
        tetrahedra.push_back(tet);
    }
    
    CDTT.resize(tetrahedra.size(), 4);
    for (int i = 0; i < tetrahedra.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            CDTT(i, j) = tetrahedra[i][j];
        }
    }
    
    // Extract ALL faces from tetrahedra
    std::set<std::array<int, 3>> unique_faces;
    
    for (const auto& tet : tetrahedra) {
        // Each tetrahedron has 4 faces
        std::array<std::array<int, 3>, 4> faces = {{
            {tet[0], tet[1], tet[2]},  // opposite to vertex 3
            {tet[0], tet[1], tet[3]},  // opposite to vertex 2
            {tet[0], tet[2], tet[3]},  // opposite to vertex 1
            {tet[1], tet[2], tet[3]}   // opposite to vertex 0
        }};
        
        for (auto& face : faces) {
            // Sort vertices to ensure consistent representation
            std::sort(face.begin(), face.end());
            unique_faces.insert(face);
        }
    }
    
    // Convert to matrix
    CDTF.resize(unique_faces.size(), 3);
    int face_idx = 0;
    for (const auto& face : unique_faces) {
        for (int j = 0; j < 3; ++j) {
            CDTF(face_idx, j) = face[j];
        }
        face_idx++;
    }
    
    constrained_delaunay = TetMesh(CDTV, CDTT, CDTF);
}

void update_view(igl::opengl::glfw::Viewer& viewer)
{
  viewer.data().clear();
  viewer.data().set_face_based(true);


  TetMesh* mesh;
  switch (display) {
    case DISPLAY_INPUT:
      viewer.data().set_mesh(V,F);
      viewer.data().set_texture(R,G,B,A);
      viewer.data().use_matcap = true;
      mesh = &tetrahedra;
      break;
    case DISPLAY_TETGEN:
      mesh = &tetrahedra;
      break;
    case DISPLAY_DELAUNAY:
      mesh = &delaunay;
      break;
    case DISPLAY_CONSTRAINED_DELAUNAY:
      mesh = &constrained_delaunay;
      break;
  }

  Eigen::VectorXd colors;
  switch (colorMap) {
    case COLORMAP_volumes:
      colors = mesh->volumes;
      break;
    case COLORMAP_av_ratio:
      colors = mesh->av_ratio;
      break;
    case COLORMAP_in_circum_ratio:
      colors = mesh->in_circum_ratio;
      break;
    case COLORMAP_aspect_ratio:
      colors = mesh->aspect_ratios;
      break;
    case COLORMAP_dihedral_angles:
      colors = mesh->dihedral_angles;
      break;
    case COLORMAP_neighbors:
      colors = mesh->max_vertex_neigbors;
      break;
    case COLORMAP_delaunay:
      colors = mesh->is_delaunay;
      break;
  }

  if(display != DISPLAY_INPUT){
    mesh->slice(slice_t, filter_t, colors, dF, dV, dC);
    viewer.data().set_mesh(dV, dF);
    viewer.data().set_colors(dC);
    igl::sort(colors, 1, true, dHistogram);
  }
}


//bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
//{
//  std::cout << "Input: " << "0x" << std::hex << static_cast<unsigned int>(static_cast<unsigned char>(key)) << std::endl;
//
//
//  if (key == 0x51) {
//    display = DISPLAY_INPUT;
//    update_view(viewer);
//  }
//  if (key == 0x57) {
//    display = DISPLAY_TETGEN;
//    update_view(viewer);
//  }
//  if (key == 0x45) {
//    display = DISPLAY_DELAUNAY;
//    update_view(viewer);
//  }
//  if (key == 0x52) {
//    display = DISPLAY_CONSTRAINED_DELAUNAY;
//    update_view(viewer);
//  }
//  if (key >= '1' && key <= '9'){
//    slice_t = double((key - '1')+1) / 9.0;
//    display_tetrahedra(viewer);
//    display_delaunay(viewer);
//    display_constrained_delaunay(viewer);
//  }
//  return false;
//}


igl::opengl::glfw::Viewer viewer;
igl::opengl::glfw::imgui::ImGuiPlugin plugin;
igl::opengl::glfw::imgui::ImGuiMenu menu;

const double p_min = 0.0;
const double p_max = 1.0;

static float getter(void* data, int idx) {
    double* plot_data = (double*)(double *)data;
    return (float) plot_data[idx];
};

void draw_menu() {
  // Define next window position + size
  ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(240, 560), ImGuiCond_FirstUseEver);
  ImGui::Begin(
      "Tetrahedra", nullptr,
      ImGuiWindowFlags_NoSavedSettings
  );

  if(ImGui::Button("Input mesh")){
    display = DISPLAY_INPUT;
    update_view(viewer);
  }
  ImGui::SameLine();
  if(ImGui::Button("TetGen")){
    display = DISPLAY_TETGEN;
    update_view(viewer);
  }
  ImGui::SameLine();
  if(ImGui::Button("Delaunay")){
    display = DISPLAY_DELAUNAY;
    update_view(viewer);
  }
  
  if(ImGui::Button("Constrained Delaunay")){
    display = DISPLAY_CONSTRAINED_DELAUNAY;
    update_view(viewer);
  }


  if(display != DISPLAY_INPUT) {
    TetMesh* mesh;
    switch (display) {
      case DISPLAY_INPUT:
	mesh = &tetrahedra;
	break;
      case DISPLAY_TETGEN:
	mesh = &tetrahedra;
	break;
      case DISPLAY_DELAUNAY:
	mesh = &delaunay;
	break;
      case DISPLAY_CONSTRAINED_DELAUNAY:
	mesh = &constrained_delaunay;
	break;
    }
    
    ImGui::Text("Number of vertices: %ld", mesh->TV.rows() );
    ImGui::Text("Number of tets: %ld", mesh->TT.rows() );

    if(ImGui::Button("smooth")){
	mesh->smooth(0.90);
	update_view(viewer);
    }

    static int e = 1;
    if(ImGui::RadioButton("Volumes", &e, 0)){
      colorMap = COLORMAP_volumes;
      update_view(viewer);
    }
    if(ImGui::RadioButton("A V ratio", &e, 1)){
      colorMap = COLORMAP_av_ratio;
      update_view(viewer);
    }
    if(ImGui::RadioButton("Insphere & circumsphere ratio", &e, 2)){
      colorMap = COLORMAP_in_circum_ratio;
      update_view(viewer);
    }
    if(ImGui::RadioButton("Min height & max length ratio", &e, 3)){
      colorMap = COLORMAP_aspect_ratio;
      update_view(viewer);
    }
    if(ImGui::RadioButton("Max dihedral angle", &e, 4)){
      colorMap = COLORMAP_dihedral_angles;
      update_view(viewer);
    }
    if(ImGui::RadioButton("Delaunay ###", &e, 5)){
      colorMap = COLORMAP_delaunay;
      update_view(viewer);
    }
    if(ImGui::RadioButton("Max vertex connectivity", &e, 6)){
      colorMap = COLORMAP_neighbors;
      update_view(viewer);
    }

    if(ImGui::SliderScalar("Slice ratio", ImGuiDataType_Double, &slice_t, &p_min, &p_max, "ratio = %.3f")) { 
      update_view(viewer);
    }
    if(ImGui::SliderScalar("Filter ratio", ImGuiDataType_Double, &filter_t, &p_min, &p_max, "ratio = %.3f")) {
      update_view(viewer);
    }

    ImGui::PlotHistogram("", &getter, dHistogram.data(), dHistogram.size(), 0, NULL, dHistogram.minCoeff(), dHistogram.maxCoeff(), ImVec2(0, 80.0f));
#ifdef HIP_ENABLED
    if(ImGui::Button("Smooth on GPU")) {
      using namespace std;
      using namespace Eigen;

      VectorXi prefix_sum; 
      VectorXi indexes; 
      TetMesh::csr_from_AM(mesh->AM, prefix_sum, indexes);

      MatrixXi edges; 
      TetMesh::edge_pairs_from_TT(mesh->TT, edges);

      cout << "prefix sum: " << prefix_sum << endl;
      cout << mesh->TV << endl;

      VertexProcessor::smooth_tets_naive(mesh->TV.data(), mesh->TV.rows(), edges.data(), edges.rows(), prefix_sum.data());

      cout << mesh->TV << endl;

      mesh->points_changed();
      update_view(viewer);
    }
#endif
  }
  
  ImGui::Separator();
  ImGui::Text("GPU Functions");
  
#ifdef HIP_ENABLED
  if(ImGui::Button("Scale Vertices (GPU)")) {
    std::vector<float> vertices_flat;
    for(int i = 0; i < V.rows(); i++) {
      vertices_flat.push_back(static_cast<float>(V(i, 0)));
      vertices_flat.push_back(static_cast<float>(V(i, 1)));
      vertices_flat.push_back(static_cast<float>(V(i, 2)));
    }
    
    VertexProcessor::processVertices(vertices_flat, V.rows(), 1.1f);
    
    for(int i = 0; i < V.rows(); i++) {
      V(i, 0) = vertices_flat[i * 3];
      V(i, 1) = vertices_flat[i * 3 + 1];
      V(i, 2) = vertices_flat[i * 3 + 2];
    }
    update_view(viewer);
  }
  
  if(ImGui::Button("Translate Vertices (GPU)")) {
    std::vector<float> vertices_flat;
    for(int i = 0; i < V.rows(); i++) {
      vertices_flat.push_back(static_cast<float>(V(i, 0)));
      vertices_flat.push_back(static_cast<float>(V(i, 1)));
      vertices_flat.push_back(static_cast<float>(V(i, 2)));
    }
    
    VertexProcessor::translateVertices(vertices_flat, V.rows(), 0.1f, 0.0f, 0.0f);
    
    for(int i = 0; i < V.rows(); i++) {
      V(i, 0) = vertices_flat[i * 3];
      V(i, 1) = vertices_flat[i * 3 + 1];
      V(i, 2) = vertices_flat[i * 3 + 2];
    }

    update_view(viewer);
  }
#else
  ImGui::Text("HIP not available - GPU functions disabled");
#endif
  
  ImGui::End();
}

//const std::string mesh = "../../libigl/meshes/53754.stl";
//const std::string mesh = "../../libigl/meshes/spot_triangulated.obj";
const std::string mesh = "../../libigl/meshes/tetrahedron.obj";

int main(int argc, char *argv[])
{
  using namespace std;
  using namespace Eigen;
  if(!igl::readOBJ(mesh, V, F)){
    std::cout << "Failed to load obj\n";
  }
  //Eigen::MatrixXi N;
  //ifstream stl_file(mesh);
  //if(!igl::readSTL(stl_file, V, F, N)){
  //  std::cout << "Failed to load stl\n";
  //}
  //stl_file.close();

  // Tetrahedralize the interior
  MatrixXd TV;
  MatrixXi TT;
  MatrixXi TF;


  igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);
  igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);

  tetrahedra = TetMesh(TV, TT, TF);

  // Create delaunay using cgal
  create_delaunay(V, F);
  
  // Create constrained delaunay using cgal
  create_constrained_delaunay(V, F);

  // Add matcap
  igl::stb::read_image("../../libigl/matcap/ceramic_dark.png", R,G,B,A); 

  // Attach a menu plugin
  viewer.plugins.push_back(&plugin);
  plugin.widgets.push_back(&menu);
  menu.callback_draw_custom_window = &draw_menu;

  viewer.core().background_color.setConstant(0.3f);
  //viewer.callback_key_down = &key_down;
  //key_down(viewer, '5', 0);
  update_view(viewer);
  viewer.launch();
}
