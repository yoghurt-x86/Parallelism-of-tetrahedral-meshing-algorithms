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
typedef Delaunay::Vertex_handle Vertex_handle;

// Input mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;


// Tetrahedralized interior
Eigen::MatrixXd TV;
Eigen::MatrixXi TT;
Eigen::MatrixXi TF;
Eigen::MatrixXd Bc;
Eigen::MatrixXd TTVC;
Eigen::MatrixXd C;


// tetrahedra diplay triangles
Eigen::MatrixXd dV;
Eigen::MatrixXi dF;

// Delaunay
Eigen::MatrixXd DTV;
Eigen::MatrixXi DTT;
Eigen::MatrixXi DTF;
Eigen::MatrixXd DBc;
Eigen::MatrixXd DTTVC;

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

    igl::barycenter(DTV,DTT,DBc);
}

void update_view(igl::opengl::glfw::Viewer& viewer)
{
  viewer.data().clear();
  viewer.data().set_face_based(true);

  switch (display) {
    case DISPLAY_INPUT:
      viewer.data().set_mesh(V,F);
      viewer.data().set_colors(C);
      break;
    case DISPLAY_TETGEN:
      viewer.data().set_mesh(dV, dF);
      viewer.data().set_colors(TTVC);
      break;
    case DISPLAY_DELAUNAY:
      viewer.data().set_mesh(dDV, dDF);
      viewer.data().set_colors(DTTVC);
      break;
  }

  //viewer.data().set_texture(R,G,B,A);
  //viewer.data().use_matcap = true;
}

void display_delaunay(igl::opengl::glfw::Viewer& viewer, unsigned char key)
{
  using namespace std;
  using namespace Eigen;

  if (key < '1' && key > '9')
  {
    return;
  }

  double t = double((key - '1')+1) / 9.0;

  VectorXd v = DBc.col(2).array() - DBc.col(2).minCoeff();
  v /= v.col(0).maxCoeff();

  vector<int> tet_i;

  for (unsigned i=0; i<v.size();++i)
    if (v(i) < t)
      tet_i.push_back(i);

  dDV = MatrixXd(tet_i.size()*4,3);
  dDF = MatrixXi(tet_i.size()*4,3);

  for (unsigned i=0; i<tet_i.size();++i)
  {
    dDV.row(i*4+0) = DTV.row(DTT(tet_i[i],0));
    dDV.row(i*4+1) = DTV.row(DTT(tet_i[i],1));
    dDV.row(i*4+2) = DTV.row(DTT(tet_i[i],2));
    dDV.row(i*4+3) = DTV.row(DTT(tet_i[i],3));
    dDF.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
    dDF.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
    dDF.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
    dDF.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
  }

  VectorXd volumes = VectorXd(tet_i.size());
  DTTVC = MatrixXd(dDV.rows(),3);

  for (unsigned i=0; i<tet_i.size();++i) {
    const auto a = DTV.row(DTT(tet_i[i], 0));
    const auto b = DTV.row(DTT(tet_i[i], 2)); // weird vertex order...
    const auto c = DTV.row(DTT(tet_i[i], 1));
    const auto d = DTV.row(DTT(tet_i[i], 3));

    // Calculate volumes x6 (Will be normalized)
    volumes(i) =
        (a.x()-d.x())*(b.y()-d.y())*(c.z()-d.z())
      + (b.x()-d.x())*(c.y()-d.y())*(a.z()-d.z())
      + (c.x()-d.x())*(a.y()-d.y())*(b.z()-d.z())
      - (c.x()-d.x())*(b.y()-d.y())*(a.z()-d.z())
      - (b.x()-d.x())*(a.y()-d.y())*(c.z()-d.z())
      - (a.x()-d.x())*(c.y()-d.y())*(b.z()-d.z());
    //if (volumes(i) == NAN || volumes(i) == INFINITY || volumes(i) == -INFINITY)
    //  volumes(i) == 0.0;
  }

  //normalize volumes
  volumes.array() -= volumes.minCoeff();
  volumes /= volumes.maxCoeff();

  for (unsigned i=0; i<tet_i.size();++i) {
    DTTVC.row(i*4+0) << 1.0, volumes(i), volumes(i);
    DTTVC.row(i*4+1) << 1.0, volumes(i), volumes(i);
    DTTVC.row(i*4+2) << 1.0, volumes(i), volumes(i);
    DTTVC.row(i*4+3) << 1.0, volumes(i), volumes(i);
  }
  update_view(viewer);
}

void calculate_volumes(igl::opengl::glfw::Viewer& viewer) {
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

  cout << v << endl;

  vector<int> tet_i;

  for (unsigned i=0; i<v.size();++i)
    if (v(i) < t)
      tet_i.push_back(i);

  dV = MatrixXd(tet_i.size()*4,3);
  dF = MatrixXi(tet_i.size()*4,3);

  for (unsigned i=0; i<tet_i.size();++i)
  {
    dV.row(i*4+0) = TV.row(TT(tet_i[i],0));
    dV.row(i*4+1) = TV.row(TT(tet_i[i],1));
    dV.row(i*4+2) = TV.row(TT(tet_i[i],2));
    dV.row(i*4+3) = TV.row(TT(tet_i[i],3));
    dF.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
    dF.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
    dF.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
    dF.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
  }

  VectorXd volumes = VectorXd(tet_i.size());
  TTVC = MatrixXd(dV.rows(),3);

  for (unsigned i=0; i<tet_i.size();++i) {
    const auto a = TV.row(TT(tet_i[i], 0));
    const auto b = TV.row(TT(tet_i[i], 2)); // weird vertex order...
    const auto c = TV.row(TT(tet_i[i], 1));
    const auto d = TV.row(TT(tet_i[i], 3));

    // Calculate volumes x6 (Will be normalized)
    volumes(i) =
        (a.x()-d.x())*(b.y()-d.y())*(c.z()-d.z())
      + (b.x()-d.x())*(c.y()-d.y())*(a.z()-d.z())
      + (c.x()-d.x())*(a.y()-d.y())*(b.z()-d.z())
      - (c.x()-d.x())*(b.y()-d.y())*(a.z()-d.z())
      - (b.x()-d.x())*(a.y()-d.y())*(c.z()-d.z())
      - (a.x()-d.x())*(c.y()-d.y())*(b.z()-d.z());
    //if (volumes(i) == NAN || volumes(i) == INFINITY || volumes(i) == -INFINITY)
    //  volumes(i) == 0.0;
  }

  //normalize volumes
  volumes.array() -= volumes.minCoeff();
  volumes /= volumes.maxCoeff();

  for (unsigned i=0; i<tet_i.size();++i) {
    TTVC.row(i*4+0) << 1.0, volumes(i), volumes(i);
    TTVC.row(i*4+1) << 1.0, volumes(i), volumes(i);
    TTVC.row(i*4+2) << 1.0, volumes(i), volumes(i);
    TTVC.row(i*4+3) << 1.0, volumes(i), volumes(i);
  }

  update_view(viewer);
}


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  std::cout << "Input: " << "0x" << std::hex << static_cast<unsigned int>(static_cast<unsigned char>(key)) << std::endl;


  if (key == 0x51) {
    display = DISPLAY_INPUT;
    update_view(viewer);
  }
  if (key == 0x57) {
    display = DISPLAY_TETGEN;
    update_view(viewer);
  }
  if (key == 0x45) {
    display = DISPLAY_DELAUNAY;
    update_view(viewer);
  }
  if (key >= '1' && key <= '9'){
    display_tetrahedra(viewer, key);
    display_delaunay(viewer, key);
  }
  return false;
}

const std::string mesh = "../meshes/spot_triangulated.obj";
//const std::string mesh = "../meshes/tetrahedron.obj";

int main(int argc, char *argv[])
{
  using namespace std;
  if(!igl::readOBJ(mesh, V, F)){
    std::cout << "Failed to load obj\n";
  }

  C =
    (V.rowwise()            - V.colwise().minCoeff()).array().rowwise()/
    (V.colwise().maxCoeff() - V.colwise().minCoeff()).array();

  cout << "V SIZE: " << std::to_string(V.size()) << endl;
  cout << "C SIZE: " << std::to_string(C.size()) << endl;


  // Tetrahedralize the interior
  igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);
  std::cout << TV.size() << std::endl;


  // Compute barycenters
  igl::barycenter(TV,TT,Bc);

  // Create delaunay using cgal
  create_delaunay();
  std::cout << DTV.size() << std::endl;
  std::cout << DTF.size() << std::endl;

  // Add matcap
  igl::stb::read_image("../matcap/ceramic_dark.png", R,G,B,A); 

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.core().background_color.setConstant(0.3f);
  viewer.callback_key_down = &key_down;
  key_down(viewer, '5', 0);
  viewer.launch();
}
