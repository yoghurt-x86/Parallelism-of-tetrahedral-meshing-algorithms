import TetMesh;
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/readOBJ.h>
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
#include <vector>
#include <map>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;

// Input mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;

// Tetrahedralized interior
//Eigen::MatrixXd TV;
//Eigen::MatrixXi TT;
//Eigen::MatrixXi TF;
//Eigen::MatrixXd Bc;
Eigen::MatrixXd TTVC;
//Eigen::MatrixXd AM;
//Eigen::VectorXd T_volumes;
//Eigen::VectorXd T_av_ratio;
//Eigen::VectorXd T_in_circum_ratio;
//Eigen::VectorXd T_aspect_ratio;
//Eigen::VectorXd T_dihedral_angles;

// tetrahedra diplay triangles
Eigen::MatrixXd dV;
Eigen::MatrixXi dF;
Eigen::VectorXd dHistogram;

// Delaunay
//Eigen::MatrixXd DTV;
//Eigen::MatrixXi DTT;
//Eigen::MatrixXi DTF;
//Eigen::MatrixXd DBc;
Eigen::MatrixXd DTTVC;
//Eigen::MatrixXd DAM;
//Eigen::VectorXd DT_av_ratio;
//Eigen::VectorXd DT_volumes;

// Delaunay diplay triangles
Eigen::MatrixXd dDV;
Eigen::MatrixXi dDF;
Eigen::VectorXd dDHistogram;

// Constrained Delaunay
//Eigen::MatrixXd CDTV;
//Eigen::MatrixXi CDTT;
//Eigen::MatrixXi CDTF;
//Eigen::MatrixXd CDBc;
Eigen::MatrixXd CDTTVC;
//Eigen::MatrixXd CDAM;
//Eigen::VectorXd CDT_av_ratio;
//Eigen::VectorXd CDT_volumes;

// Constrained Delaunay diplay triangles
Eigen::MatrixXd dCDV;
Eigen::MatrixXi dCDF;
Eigen::VectorXd dCDHistogram;

class TetMesh {
private:
    void points_changed();

    static void adjacency(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::MatrixXd &AM);
    static void compute_aspect_ratios(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out);
    static void compute_volumes(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out);
    static void area_volume_ratio(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out);
    static void insphere_to_circumsphere(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out);
    static void compute_dihedral_angles(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out);
    static void count_neighbors(const Eigen::MatrixXi& AM, Eigen::VectorXi &out);
public:
    // Data
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TF;
    Eigen::MatrixXd C;
    Eigen::MatrixXd AM;
    Eigen::VectorXd volumes;
    Eigen::VectorXd av_ratio;
    Eigen::VectorXd in_circum_ratio;
    Eigen::VectorXd aspect_ratios;
    Eigen::VectorXd dihedral_angles;
    TetMesh(); // unitialized
    TetMesh(Eigen::MatrixXd TV, Eigen::MatrixXi TT, Eigen::MatrixXi TF);
    void slice(double slice_t, double ratio_t, const Eigen::VectorXd _colors, Eigen::MatrixXi &dF, Eigen::MatrixXd &dV, Eigen::MatrixXd &C);
};

TetMesh::TetMesh() {}

void TetMesh::points_changed() {
    adjacency(TT, TV, AM);
    compute_volumes(TT,TV, volumes);
    area_volume_ratio(TT, TV,volumes, av_ratio);
    insphere_to_circumsphere(TT, TV, volumes, in_circum_ratio);
    compute_aspect_ratios(TT, TV, volumes, aspect_ratios);
    compute_dihedral_angles(TT, TV, dihedral_angles);
}

TetMesh::TetMesh(Eigen::MatrixXd _TV, Eigen::MatrixXi _TT, Eigen::MatrixXi _TF) {
    using namespace std;

    TV = _TV;
    TT = _TT;
    TF = _TF;

    points_changed();

    cout << "Vertices: " << TV.rows() << endl;
    cout << "Faces: " << TF.rows() << endl;
    cout << "Tets: " << TT.rows() << endl;
    cout << "Tets max: " << TT.maxCoeff() << endl;
}


void TetMesh::adjacency(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::MatrixXd &AM){
    using namespace std;
    using namespace Eigen;

    AM.resize(TV.rows(), TV.rows());

    for (unsigned i=0; i<TT.rows(); ++i) {
        for (unsigned j=0; j<4; j++) {
            for (unsigned k=0; k<4; k++) {
                if (k != j)
                    AM(TT(i,j), TT(i,k)) = 1;
            }
        }
    }
}

void TetMesh::count_neighbors(const Eigen::MatrixXi& AM, Eigen::VectorXi &out) {
    out.resize(AM.rows());
    for (unsigned i=0; i<AM.rows();++i) {
      unsigned count = 0;
      for (unsigned j=0; j<AM.cols();++j) {
	if (AM(i,j) != 0) {
	  count++;
	}
      }
      out(i) = count;
    }
}


// See "What is a good finite element" Jonathan Richard Shewchuk p. 61
void TetMesh::compute_volumes(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out) {
    out.resize(TT.rows());
    //C.resize( dV.rows(),3);

    for (unsigned i=0; i<TT.rows();++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        // Calculate volumes x6 (Will be normalized)
        const auto V =
            (a.x()-d.x())*(b.y()-d.y())*(c.z()-d.z())
          + (b.x()-d.x())*(c.y()-d.y())*(a.z()-d.z())
          + (c.x()-d.x())*(a.y()-d.y())*(b.z()-d.z())
          - (c.x()-d.x())*(b.y()-d.y())*(a.z()-d.z())
          - (b.x()-d.x())*(a.y()-d.y())*(c.z()-d.z())
          - (a.x()-d.x())*(c.y()-d.y())*(b.z()-d.z());

        out(i) = V * (1.0/6.0);

        //if (volumes(i) == NAN || volumes(i) == INFINITY || volumes(i) == -INFINITY)
        //  volumes(i) == 0.0;
    }
}


// See "What is a good finite element" Jonathan Richard Shewchuk p. 54
void TetMesh::area_volume_ratio(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out) {
  // This returns 4A^2
  auto area = [&] (auto a, auto b, auto c) {
    auto yz = ((a.y()-c.y())*(b.z()-c.z()))-((b.y()-c.y())*(a.z()-c.z()));
    auto zx = ((a.z()-c.z())*(b.x()-c.x()))-((b.z()-c.z())*(a.x()-c.x()));
    auto xy = ((a.x()-c.x())*(b.y()-c.y()))-((b.x()-c.x())*(a.y()-c.y()));
    return yz*yz + zx*zx + xy*xy;
  };

  out.resize(TT.rows());
  for (unsigned i=0; i<TT.rows();++i) {
    const Eigen::Block<const Eigen::Matrix<double, -1, -1>, 1> a = TV.row(TT(i, 0));
    const auto b = TV.row(TT(i, 2)); // tetgen uses a different vertex order than Shewchuk...
    const auto c = TV.row(TT(i, 1));
    const auto d = TV.row(TT(i, 3));

    double a1 = area(d,b,c);
    double a2 = area(a,d,c);
    double a3 = area(a,d,b);
    double a4 = area(a,b,c);

    double area_sum = (a1 + a2 + a3 + a4) * 0.25 * 0.75;
    out(i) = (volumes(i) / area_sum) * 2.61505662862; //3^(7/8)
  }
}

void TetMesh::insphere_to_circumsphere(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out) {
    using namespace Eigen;
    using namespace std;

    auto area = [&] (auto a, auto b, auto c) {
        auto yz = ((a.y()-c.y())*(b.z()-c.z()))-((b.y()-c.y())*(a.z()-c.z()));
        auto zx = ((a.z()-c.z())*(b.x()-c.x()))-((b.z()-c.z())*(a.x()-c.x()));
        auto xy = ((a.x()-c.x())*(b.y()-c.y()))-((b.x()-c.x())*(a.y()-c.y()));
        return std::sqrt(yz*yz + zx*zx + xy*xy)*0.5;
    };

    out.resize(TT.rows());

    for (unsigned i=0; i<TT.rows();++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        const Vector3d t = (a-d);
        const Vector3d u = (b-d);
        const Vector3d v = (c-d);

        const double tabs = t.dot(t);
        const double uabs = u.dot(u);
        const double vabs = v.dot(v);

        double Z = ((tabs * u).cross(v) + (uabs * v).cross(t) + (vabs*t).cross(u)).norm();

        double a1 = area(d,b,c);
        double a2 = area(a,d,c);
        double a3 = area(a,d,b);
        double a4 = area(a,b,c);
        double A = a1+a2+a3+a4;

        out(i) = 108.0 * ((volumes(i)*volumes(i)) / (Z*A));
    }
}

void TetMesh::compute_aspect_ratios(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out) {
    using namespace Eigen;
    using namespace std;

    out.resize(TT.rows());

    for (unsigned i=0; i<TT.rows();++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        Eigen::Matrix<double, Eigen::Dynamic, 3> ls(6, 3);
        ls.row(0) = b-a;
        ls.row(1) = a-c;
        ls.row(2) = a-d;
        ls.row(3) = c-b;
        ls.row(4) = b-d;
        ls.row(5) = c-d;

        MatrixXd norms = ls.rowwise().norm();
        const double lmax = norms.maxCoeff();

        MatrixXd crosses = MatrixXd(15,3);
        unsigned int k = 0;
        for(unsigned n=0; n<6; ++n){
          for(unsigned m=n+1; m<6; ++m) {
              crosses.row(k) = ls.row(n).cross(ls.row(m));
              k++;
           }
        }
        cout << k << endl;

        auto max_crosswx = crosses.rowwise().norm().array().maxCoeff();

        out(i) = (volumes(i) / max_crosswx) * 6.0 * M_SQRT2;
    }
}

void TetMesh::compute_dihedral_angles(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out) {
    using namespace Eigen;
    using namespace std;

    out.resize(TT.rows());

    for (unsigned i=0; i<TT.rows();++i) {
        const Vector3d a = TV.row(TT(i, 0));
        const Vector3d b = TV.row(TT(i, 2)); // weird vertex order...
        const Vector3d c = TV.row(TT(i, 1));
        const Vector3d d = TV.row(TT(i, 3));

        Eigen::Matrix<double, Eigen::Dynamic, 3> normal(4, 3);
        normal.row(0) = (a - c).cross(d-c).normalized();
        normal.row(1) = (c - b).cross(d-b).normalized();
        normal.row(2) = (b - a).cross(d-a).normalized();
        normal.row(3) = (a - b).cross(c-b).normalized();

        VectorXd cartesian_normals(6);
        unsigned int k = 0;
        for(unsigned n=0; n<4; ++n){
            for(unsigned m=n+1; m<4; ++m) {
                auto cos_angle = clamp(normal.row(n).dot(normal.row(m)), -1.0, 1.0);
                cartesian_normals(k) = M_PI - std::acos(cos_angle);
	            k++;
	        }
        }
        out(i) = cartesian_normals.maxCoeff();
    }
}

void TetMesh::slice(double slice_t, double filter_t, const Eigen::VectorXd _colors, Eigen::MatrixXi &dF, Eigen::MatrixXd &dV, Eigen::MatrixXd &C){
  using namespace std;
  using namespace Eigen;

  // Compute barycenters
  MatrixXd Bc;
  igl::barycenter(TV,TT,Bc);

  VectorXd v = Bc.col(2).array() - Bc.col(2).minCoeff();
  v /= v.col(0).maxCoeff();

  //normalize volumes
  VectorXd color = _colors;
  color.array() -= color.minCoeff();
  color /= color.maxCoeff();

  vector<int> sorted_i(v.size());
  // Initialize with indices 0, 1, 2, ..., n-1
  std::iota(sorted_i.begin(), sorted_i.end(), 0);

  // Sort indices based on the values in v
  std::sort(sorted_i.begin(), sorted_i.end(), [&v](int a, int b) {
      return v(a) < v(b);
  });

  vector<int> tet_i;
  for (int idx : sorted_i) {
      if (v(idx) < slice_t && color(idx) > filter_t) {
          tet_i.push_back(idx);
      }
  }
  // make sure it's not empty
  if (tet_i.empty()) {
      tet_i.push_back(sorted_i[0]);
  }

  dV.resize(tet_i.size()*4,3);
  dF.resize(tet_i.size()*4,3);
  VectorXd dColors = VectorXd(dV.rows());
  C.resize( dV.rows(),3);

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

    dColors(i*4+0) = color(tet_i[i]);
    dColors(i*4+1) = color(tet_i[i]);
    dColors(i*4+2) = color(tet_i[i]);
    dColors(i*4+3) = color(tet_i[i]);
  }

  igl::jet(dColors, false, C);

  //for (unsigned i=0; i<tet_i.size();++i) {
  //  const unsigned j = tet_i[i];
  //  C.row(i*4+0) << 1.0, color(j), color(j);
  //  C.row(i*4+1) << 1.0, color(j), color(j);
  //  C.row(i*4+2) << 1.0, color(j), color(j);
  //  C.row(i*4+3) << 1.0, color(j), color(j);
  //}
}

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

Display display = DISPLAY_INPUT;
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

  switch (display) {
    case DISPLAY_INPUT:
      viewer.data().set_mesh(V,F);
      viewer.data().set_texture(R,G,B,A);
      viewer.data().use_matcap = true;
      break;
    case DISPLAY_TETGEN:
      viewer.data().set_mesh(dV, dF);
      viewer.data().set_colors(TTVC);
      break;
    case DISPLAY_DELAUNAY:
      viewer.data().set_mesh(dDV, dDF);
      viewer.data().set_colors(DTTVC);
      break;
    case DISPLAY_CONSTRAINED_DELAUNAY:
      viewer.data().set_mesh(dCDV, dCDF);
      viewer.data().set_colors(CDTTVC);
      break;
  }

  //viewer.data().set_texture(R,G,B,A);
  //viewer.data().use_matcap = true;
}

void smooth_tetrahedra(const double t, const Eigen::MatrixXi& TT,  Eigen::MatrixXd& TV, const Eigen::MatrixXd &AM, Eigen::VectorXd volumes, Eigen::VectorXd av_ratio) {
    using namespace std;
    using namespace Eigen;

    MatrixXd result = MatrixXd(TV.rows(), 3);

    for (unsigned i=0; i<TV.rows(); ++i) {
        unsigned count = 0;
        result.row(i) << 0,0,0;
        for (unsigned j=0; j<TV.rows(); ++j) {
            if (AM(i, j) == 1) { // If adjacen
                count++;
            }
        }

        double ratio = 1.0 / double(count);

        for (unsigned j=0; j<TV.rows(); ++j) {
            if (AM(i, j)) { // If adjacent
                result.row(i) += (TV.row(j) * ratio);
            }
        }

        //result.row(i) /= count;
        result.row(i) = (result.row(i) * (1-t)) + (TV.row(i) * t);
    }

    TV = result;
    //compute_volumes(TT, TV, volumes);
    //area_volume_ratio(TT, TV, volumes, av_ratio);
}


void display_delaunay(igl::opengl::glfw::Viewer& viewer)
{
  using namespace std;
  using namespace Eigen;

  delaunay.slice(slice_t, filter_t, delaunay.av_ratio, dDF, dDV, DTTVC);

  igl::sort(delaunay.av_ratio, 1, true, dDHistogram);

  update_view(viewer);
}

void display_constrained_delaunay(igl::opengl::glfw::Viewer& viewer)
{
  using namespace std;
  using namespace Eigen;

  constrained_delaunay.slice(slice_t, filter_t, constrained_delaunay.av_ratio, dCDF, dCDV, CDTTVC);
  update_view(viewer);
}

void display_tetrahedra(igl::opengl::glfw::Viewer& viewer)
{
  using namespace std;
  using namespace Eigen;

  tetrahedra.slice(slice_t, filter_t, tetrahedra.aspect_ratios, dF, dV, TTVC);
  igl::sort(tetrahedra.aspect_ratios, 1, true, dHistogram);
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
  if (key == 0x52) {
    display = DISPLAY_CONSTRAINED_DELAUNAY;
    update_view(viewer);
  }
  if (key >= '1' && key <= '9'){
    slice_t = double((key - '1')+1) / 9.0;
    display_tetrahedra(viewer);
    display_delaunay(viewer);
    display_constrained_delaunay(viewer);
  }
  return false;
}


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
  ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
  ImGui::Begin(
      "New Window", nullptr,
      ImGuiWindowFlags_NoSavedSettings
  );


  //if(ImGui::DragScalar("double", ImGuiDataType_Double, &slice_t, 0.001, &p_min, &p_max, "%.4f")){

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

  if(ImGui::Button("smooth")){
      //smooth_tetrahedra(0.5, TT, TV, AM, T_av_ratio, T_volumes);
      //smooth_tetrahedra(0.5, DTT, DTV, DAM, DT_av_ratio, T_volumes);
      display_tetrahedra(viewer);
      display_delaunay(viewer);
  }

  if(display != DISPLAY_INPUT) {
    if(ImGui::SliderScalar("Slice ratio", ImGuiDataType_Double, &slice_t, &p_min, &p_max, "ratio = %.3f")) { 
      display_tetrahedra(viewer);
      display_delaunay(viewer);
      display_constrained_delaunay(viewer);
    }
    if(ImGui::SliderScalar("Filter ratio", ImGuiDataType_Double, &filter_t, &p_min, &p_max, "ratio = %.3f")) {
      display_tetrahedra(viewer);
      display_delaunay(viewer);
      display_constrained_delaunay(viewer);
    }

    ImGui::PlotHistogram("Histogram", &getter, dHistogram.data(), dHistogram.size(), 0, NULL, dHistogram.minCoeff(), dHistogram.maxCoeff(), ImVec2(0, 80.0f));
  }
  ImGui::End();
}

const std::string mesh = "../../libigl/meshes/spot_triangulated.obj";
//const std::string mesh = "../meshes/tetrahedron.obj";

int main(int argc, char *argv[])
{
  using namespace std;
  using namespace Eigen;
  if(!igl::readOBJ(mesh, V, F)){
    std::cout << "Failed to load obj\n";
  }

  // Tetrahedralize the interior
  MatrixXd TV;
  MatrixXi TT;
  MatrixXi TF;
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
  viewer.callback_key_down = &key_down;
  key_down(viewer, '5', 0);
  viewer.launch();
}
