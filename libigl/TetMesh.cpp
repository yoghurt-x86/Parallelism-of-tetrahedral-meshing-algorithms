#include "TetMesh.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <array>
#include <igl/barycenter.h>
#include <igl/jet.h>
#include <igl/cumsum.h>

TetMesh::TetMesh() {}

void TetMesh::points_changed() {
    adjacency(TT, TV, AM);
    compute_volumes(TT,TV, volumes);
    area_volume_ratio(TT, TV,volumes, av_ratio);
    insphere_to_circumsphere(TT, TV, volumes, in_circum_ratio);
    compute_aspect_ratios(TT, TV, volumes, aspect_ratios);
    compute_dihedral_angles(TT, TV, dihedral_angles);
    count_neighbors(TT, AM, max_vertex_neigbors);
    compute_is_delaunay(TT, TV, AM, is_delaunay);
}

TetMesh::TetMesh(const Eigen::MatrixXd& _TV, const Eigen::MatrixXi& _TT, const Eigen::MatrixXi& _TF) {
    using namespace std;

    TV = _TV;
    TT = _TT;
    TF = _TF;

    points_changed();

}

void TetMesh::adjacency(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::MatrixXi &AM){
    using namespace std;
    using namespace Eigen;

    AM.setZero(TV.rows(), TV.rows());

    for (unsigned i=0; i<TT.rows(); ++i) {
        for (unsigned j=0; j<4; j++) {
            for (unsigned k=0; k<4; k++) {
                if (k != j)
                    AM(TT(i,j), TT(i,k)) = 1;
            }
        }
    }
}
void TetMesh::csr_from_AM(const Eigen::MatrixXi &AM, Eigen::VectorXi &prefix_sum, Eigen::VectorXi &V) {
    using namespace std;
    using namespace Eigen;

    auto sums = AM.rowwise().sum();
    igl::cumsum(sums, 1, prefix_sum);

    // Set the size of idxs to the number of edges
    V.setZero((prefix_sum(prefix_sum.size()-1)) * 2); 
  
    auto idx = 0;
    for(unsigned i=0; i<AM.rows(); ++i) {
      for(unsigned j=0; j<AM.cols(); ++j) {
	if (AM(i,j) != 0) {
	  V(idx) = i;
	  idx++;
	  V(idx) = j;
	  idx++;
	}
      }
    }
}


void TetMesh::edge_pairs_from_TT(const Eigen::MatrixXi &TT, Eigen::MatrixXi &edges) {
    using namespace std;
    using namespace Eigen;

    // Use set to store unique edges (automatically sorted)
    set<pair<int, int>> edge_set;

    // Extract edges from each tetrahedron
    for (int i = 0; i < TT.rows(); ++i) {
	//vector<int> tet = {TT(i,0), TT(i,1), TT(i,2), TT(i,3)};
	auto tet = TT.row(i);

	// Generate all 6 edges of the tetrahedron
	for (int j = 0; j < 4; ++j) {
	    for (int k = j + 1; k < 4; ++k) {
		int v1 = tet(j), v2 = tet(k);
		if (v1 > v2) swap(v1, v2); // Ensure consistent ordering
		edge_set.insert({v1, v2});
	    }
	}
    }

    // Store edges as vertex pairs
    edges.resize(edge_set.size(), 2);
    int idx = 0;
    for (const auto& edge : edge_set) {
	edges(idx, 0) = edge.first;
	edges(idx, 1) = edge.second;
	idx++;
    }
}

//compressed sparse row of vertices
//void TetMesh::csr(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, Eigen::VectorXi &prefix_sum, Eigen::VectorXi &idxs) {
//    using namespace std;
//    using namespace Eigen;
//}

void TetMesh::count_neighbors(const Eigen::MatrixXi TT, const Eigen::MatrixXi& AM, Eigen::VectorXd &out) {
    using namespace Eigen;

    VectorXi neighbors(AM.rows());
    for (unsigned i=0; i<AM.rows();++i) {
      unsigned count = 0;
      for (unsigned j=0; j<AM.cols();++j) {
	if (AM(i,j) != 0) {
	  count++;
	}
      }
      neighbors(i) = count;
    }

    out.resize(TT.rows());
    for(unsigned i=0; i<TT.rows(); ++i){
      Vector4d res; 
      res(0) = (double) neighbors(TT(i,0));
      res(1) = (double) neighbors(TT(i,1));
      res(2) = (double) neighbors(TT(i,2));
      res(3) = (double) neighbors(TT(i,3));
      out(i) = res.maxCoeff();
    } 
}

// See "What is a good finite element" Jonathan Richard Shewchuk p. 61
void TetMesh::compute_volumes(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out) {
    using namespace Eigen;

    out.resize(TT.rows());

    for (unsigned i=0; i<TT.rows();++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));


        // Calculate volumes x6 
   
	Matrix3d det;
	det << (a.x()-d.x()), (b.x()-d.x()), (c.x()-d.x())
	     , (a.y()-d.y()), (b.y()-d.y()), (c.y()-d.y())
	     , (a.z()-d.z()), (b.z()-d.z()), (c.z()-d.z());

	const auto V = det.determinant() * (1.0/6.0);
  
        //const auto V =
        //    (a.x()-d.x())*(b.y()-d.y())*(c.z()-d.z())
        //  + (b.x()-d.x())*(c.y()-d.y())*(a.z()-d.z())
        //  + (c.x()-d.x())*(a.y()-d.y())*(b.z()-d.z())
        //  - (c.x()-d.x())*(b.y()-d.y())*(a.z()-d.z())
        //  - (b.x()-d.x())*(a.y()-d.y())*(c.z()-d.z())
        //  - (a.x()-d.x())*(c.y()-d.y())*(b.z()-d.z());

        out(i) = V ;
    }
}


void TetMesh::compute_is_delaunay(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::MatrixXi &AM, Eigen::VectorXd &out) {
    using namespace Eigen;
    auto in_sphere = [&] (auto a, auto b, auto c, auto d, auto p) {
        const double ax = a.x() - p.x();
	const double ay = a.y() - p.y();
	const double az = a.z() - p.z();
	
	const double bx = b.x() - p.x();
	const double by = b.y() - p.y();
	const double bz = b.z() - p.z();
	
	const double cx = c.x() - p.x();
	const double cy = c.y() - p.y();
	const double cz = c.z() - p.z();
	
	const double dx = d.x() - p.x();
	const double dy = d.y() - p.y();
	const double dz = d.z() - p.z();
	
	const double a_sqr = ax*ax + ay*ay + az*az;
	const double b_sqr = bx*bx + by*by + bz*bz;
	const double c_sqr = cx*cx + cy*cy + cz*cz;
	const double d_sqr = dx*dx + dy*dy + dz*dz;
	
	Matrix<double, 4, 4> det;
	det << ax, ay, az, a_sqr,
	       bx, by, bz, b_sqr,
	       cx, cy, cz, c_sqr,
	       dx, dy, dz, d_sqr;
	
	return det.determinant();
    };
    out.resize(TT.rows());

    for(unsigned i=0; i<TT.rows();++i) {
	const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

	auto flag = 1e-13;
	for(unsigned j=0; j<TV.rows();++j) {
	    const auto p = TV.row(j);
	    if ( j != TT(i,0) &&
		 j != TT(i,1) &&
		 j != TT(i,2) &&
		 j != TT(i,3)) {
		auto res = in_sphere(a, b, c, d, p);
		if (res > 1e-14){
		    flag = res;
		}
	    }
	}
	out(i) = flag;
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
}


void TetMesh::smooth(const double t) {
    using namespace std;
    using namespace Eigen;

    MatrixXd result = TV;

    for (unsigned k=0; k<25;++k){
	for (unsigned i=0; i<result.rows(); ++i) {

	    Vector3d p(0.0,0.0,0.0); 
	    unsigned count = 0;

	    for (unsigned j=0; j<result.rows(); ++j) {
		if (AM(i, j) == 1) { // If adjacen
		    count++;
		}
	    }

	    double ratio = 1.0 / double(count);

	    for (unsigned j=0; j<result.rows(); ++j) {
		if (AM(i, j)) { // If adjacent
		    p += (result.row(j) * ratio);
		}
	    }
	    p *= 1-t;
	    p += result.row(i) * t;

	    //result.row(i) /= count;
	    result.row(i) = p;
	}
    }

    TV = result;

    this->points_changed();
}
