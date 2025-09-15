#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/copyleft/cgal/CGAL_includes.hpp>
#include <igl/barycenter.h>
#include <igl/stb/read_image.h>
#include <iostream>
#include <vector>
#include <map>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef Delaunay::Point Point;

// Input mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;


// Tetrahedralized interior
Eigen::MatrixXd TV;
Eigen::MatrixXi TT;
Eigen::MatrixXi TF;
Eigen::MatrixXd Bc;

// tetrahedra diplay triangles
Eigen::MatrixXd dV;
Eigen::MatrixXi dF;

// Delaunay
Eigen::MatrixXd DTV;
Eigen::MatrixXi DTT;
Eigen::MatrixXd DBc;

// Delaunay diplay triangles
Eigen::MatrixXd dDV;
Eigen::MatrixXi dDF;

// matcap texture
Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;

enum Display {
  DISPLAY_INPUT,
  DISPLAY_TETGEN,
  DISPLAY_DELAUNAY,
};

Display display = DISPLAY_INPUT;

void create_delaunay() {
  // Create 3D Delaunay triangulation using CGAL
  Delaunay dt;
  std::vector<Point> points;

  // Convert Eigen vertices to CGAL points
  for(int i = 0; i < V.rows(); i++) {
    points.push_back(Point(V(i,0), V(i,1), V(i,2)));
  }

  // Insert points into Delaunay triangulation
  dt.insert(points.begin(), points.end());

  // Extract tetrahedra
  std::vector<std::array<int, 4>> tets;
  std::map<Delaunay::Vertex_handle, int> vertex_map;

  // Create vertex mapping
  int vertex_idx = 0;
  for(auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
    vertex_map[vit] = vertex_idx++;
  }

  // Extract tetrahedra indices
  for(auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
    std::array<int, 4> tet;
    for(int j = 0; j < 4; j++) {
      tet[j] = vertex_map[cit->vertex(j)];
    }
    tets.push_back(tet);
  }

  // Convert back to Eigen matrices
  DTV = V; // Vertices remain the same
  DTT.resize(tets.size(), 4);
  for(int i = 0; i < tets.size(); i++) {
    for(int j = 0; j < 4; j++) {
      DTT(i, j) = tets[i][j];
    }
  }

}

void update_mesh(igl::opengl::glfw::Viewer& viewer)
{
  viewer.data().clear();

  switch (display) {
    case DISPLAY_INPUT:
      viewer.data().set_mesh(V,F);
      break;
    case DISPLAY_TETGEN:
      viewer.data().set_mesh(dV, dF);
      break;
    case DISPLAY_DELAUNAY:
      viewer.data().set_mesh(dDV, dDF);
      break;
  }

  viewer.data().set_texture(R,G,B,A);
  viewer.data().use_matcap = true;
  viewer.data().set_face_based(true);
}

void display_delaunay(igl::opengl::glfw::Viewer& viewer, unsigned char key)
{
  using namespace std;
  using namespace Eigen;

  dDV = MatrixXd(DTT.rows()*4,3);
  dDF = MatrixXi(DTT.rows()*4,3);

  for (unsigned i=0; i<DTT.rows();++i)
  {
    dDV.row(i*4+0) = DTV.row(DTT(i,0));
    dDV.row(i*4+1) = DTV.row(DTT(i,1));
    dDV.row(i*4+2) = DTV.row(DTT(i,2));
    dDV.row(i*4+3) = DTV.row(DTT(i,3));
    dDF.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
    dDF.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
    dDF.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
    dDF.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
  }
}

void display_tetrahedra(igl::opengl::glfw::Viewer& viewer, unsigned char key)
{
  using namespace std;
  using namespace Eigen;

  if (key < '1' && key > '9')
  {
    return;
  }

  double t = double((key - '1')+1) / 9.0;

  VectorXd v = Bc.col(2).array() - Bc.col(2).minCoeff();
  v /= v.col(0).maxCoeff();

  vector<int> s;

  for (unsigned i=0; i<v.size();++i)
    if (v(i) < t)
      s.push_back(i);

  dV = MatrixXd(s.size()*4,3);
  dF = MatrixXi(s.size()*4,3);

  for (unsigned i=0; i<s.size();++i)
  {
    dV.row(i*4+0) = TV.row(TT(s[i],0));
    dV.row(i*4+1) = TV.row(TT(s[i],1));
    dV.row(i*4+2) = TV.row(TT(s[i],2));
    dV.row(i*4+3) = TV.row(TT(s[i],3));
    dF.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
    dF.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
    dF.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
    dF.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
  }

  update_mesh(viewer);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  std::cout << "Input: " << "0x" << std::hex << static_cast<unsigned int>(static_cast<unsigned char>(key)) << std::endl;

  if (key == 0x51) {
    display = DISPLAY_INPUT;
    update_mesh(viewer);
  }
  if (key == 0x57) {
    display = DISPLAY_TETGEN;
    update_mesh(viewer);
  }
  if (key == 0x45) {
    display = DISPLAY_DELAUNAY;
    display_delaunay(viewer, key);
    update_mesh(viewer);
  }
  if (key >= '1' && key <= '9'){
    display_tetrahedra(viewer, key);
  }
  return false;
}

int main(int argc, char *argv[])
{
  if(!igl::readOBJ("../meshes/spot_triangulated.obj", V, F)){
    std::cout << "Failed to load obj\n";
  }

  // Tetrahedralize the interior
  igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);

  // Compute barycenters
  igl::barycenter(TV,TT,Bc);

  // Create delaunay using cgal
  create_delaunay();

  // Add matcap
  igl::stb::read_image("../matcap/ceramic_dark.png", R,G,B,A); 

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.core().background_color.setConstant(0.3f);
  viewer.callback_key_down = &key_down;
  key_down(viewer, '5', 0);
  viewer.launch();
}