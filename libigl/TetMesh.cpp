#include "TetMesh.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <set>
#include <array>
#include <map>
#include <igl/barycenter.h>
#include <igl/jet.h>
#include <igl/cumsum.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <vector>

TetMesh::TetMesh() {
}

void TetMesh::points_changed() {
    adjacency(TT, TV, AM);
    compute_volumes(TT, TV, volumes);
    area_volume_ratio(TT, TV, volumes, av_ratio);
    insphere_to_circumsphere(TT, TV, volumes, in_circum_ratio);
    compute_aspect_ratios(TT, TV, volumes, aspect_ratios);
    compute_dihedral_angles(TT, TV, dihedral_angles);
    count_neighbors(TT, AM, max_vertex_neigbors);
    compute_is_delaunay(TT, TV, AM, is_delaunay);
    //compute_boundary_flags(TV, TF, boundary_flag);
    //compute_boundary_flags(TV, TF, boundary_flag);
}

TetMesh::TetMesh(const Eigen::MatrixXd &_TV, const Eigen::MatrixXi &_TT, const Eigen::MatrixXi &_TF) {
    using namespace std;

    TV = _TV;
    TT = _TT;
    TF = _TF;

    points_changed();
}

void TetMesh::normalize_mesh(const Eigen::MatrixXd& V,
                    Eigen::MatrixXd& V_out){
    using namespace Eigen;
    Vector3d bbox_min = V.colwise().minCoeff();
    Vector3d bbox_max = V.colwise().maxCoeff();

    // Compute center and scale
    Vector3d center_out = (bbox_min + bbox_max) / 2.0;
    double scale_out = (bbox_max - bbox_min).maxCoeff();

    // Normalize
    V_out = (V.rowwise() - center_out.transpose()) / scale_out;
}

void TetMesh::adjacency(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, Eigen::MatrixXi &AM) {
    using namespace std;
    using namespace Eigen;

    AM.setZero(TV.rows(), TV.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        for (unsigned j = 0; j < 4; j++) {
            for (unsigned k = 0; k < 4; k++) {
                if (k != j)
                    AM(TT(i, j), TT(i, k)) = 1;
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
    V.setZero((prefix_sum(prefix_sum.size() - 1)) * 2);

    auto idx = 0;
    for (unsigned i = 0; i < AM.rows(); ++i) {
        for (unsigned j = 0; j < AM.cols(); ++j) {
            if (AM(i, j) != 0) {
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
    set<pair<int, int> > edge_set;

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
    for (const auto &edge: edge_set) {
        edges(idx, 0) = edge.first;
        edges(idx, 1) = edge.second;
        idx++;
    }
}

// TN is a Nx4 matrix where N is the number of tets and 4 is the four possible neighbors of a tet.
// TT is a Nx4 matrix where N is the number of tets and columns are indexes to points they share.
void TetMesh::flips(const Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN,
                    Eigen::MatrixXi &two_three_flips,
                    Eigen::MatrixXi &three_two_flips,
                    Eigen::MatrixXi &TF23
                    //Eigen::MatrixXi &TF32
                    ) {
    using namespace std;
    using namespace Eigen;

    // Use set to store unique faces (automatically sorted)
    set<pair<int, int> > face_set;

    for (unsigned i = 0; i < TN.rows(); ++i) {
        auto tet = TN.row(i);

        for (unsigned j = 0; j < 4; ++j) {
            int v1 = i;
            int v2 = tet(j);
            if (v2 == -1) continue;
            if (v1 > v2) swap(v1, v2); // Ensure consistent ordering
            face_set.insert({v1, v2});
        }
    }

    VectorXi TF23_c;
    TF23_c.setZero(TT.rows());
    TF23.setConstant(TT.rows(), 4, -1);

    two_three_flips.resize(face_set.size(), 2);
    int idx = 0;
    for (const auto &face: face_set) {
        two_three_flips(idx, 0) = face.first;
        two_three_flips(idx, 1) = face.second;
        idx++;

        TF23(face.first, TF23_c(face.first)) = idx;
        TF23(face.second, TF23_c(face.second)) = idx;
        TF23_c(face.first) += 1;
        TF23_c(face.second) += 1;
    }

    // Find edges shared by exactly three tetrahedra for three-two flips
    map<pair<int, int>, vector<int> > edge_to_tets;

    // Extract all edges from all tetrahedra and track which tets they belong to
    for (unsigned i = 0; i < TT.rows(); ++i) {
        auto tet = TT.row(i);

        // Generate all 6 edges of each tetrahedron
        for (unsigned j = 0; j < 4; ++j) {
            for (unsigned k = j + 1; k < 4; ++k) {
                int v1 = tet(j), v2 = tet(k);
                if (v1 > v2) swap(v1, v2); // Ensure consistent ordering
                edge_to_tets[{v1, v2}].push_back(i);
            }
        }
    }

    // Find edges shared by exactly 3 tetrahedra
    vector<array<int, 3> > three_two_edges;
    for (const auto &edge_entry: edge_to_tets) {
        if (edge_entry.second.size() == 3) {
            three_two_edges.push_back({
                edge_entry.second[0],
                edge_entry.second[1],
                edge_entry.second[2]
            });
        }
    }

    // Store results in output matrix
    three_two_flips.resize(three_two_edges.size(), 3);
    for (unsigned i = 0; i < three_two_edges.size(); ++i) {
        three_two_flips(i, 0) = three_two_edges[i][0];
        three_two_flips(i, 1) = three_two_edges[i][1];
        three_two_flips(i, 2) = three_two_edges[i][2];
    }

    cout << "23: \n" << two_three_flips << endl;
    cout << "32: \n" << three_two_flips << endl;
}

void TetMesh::vertex_to_TT_map(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, Eigen::VectorXi &offset_out,
                               Eigen::VectorXi &tet_index_out) {
    using namespace std;
    using namespace Eigen;

    VectorXi sums;
    sums.setZero(TV.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        sums(TT(i, 0))++;
        sums(TT(i, 1))++;
        sums(TT(i, 2))++;
        sums(TT(i, 3))++;
    }

    VectorXi prefix_sum_inclusive;
    igl::cumsum(sums, 1, prefix_sum_inclusive);
    unsigned total = prefix_sum_inclusive(prefix_sum_inclusive.size() - 1);

    //cout << "prfx sum\n" << prefix_sum_inclusive << endl << endl;

    VectorXi prefix_sum(sums.size());
    prefix_sum << 0, prefix_sum_inclusive.head(sums.size() - 1);

    //cout << prefix_sum << endl;
    offset_out = prefix_sum;

    tet_index_out.resize(total);
    assert(tet_index_out.size() == TT.rows() * 4);

    //cout << "size " << tet_index_out.size() << endl;

    for (unsigned i = 0; i < TT.rows(); ++i) {
        tet_index_out(prefix_sum(TT(i, 0))++) = i;
        tet_index_out(prefix_sum(TT(i, 1))++) = i;
        tet_index_out(prefix_sum(TT(i, 2))++) = i;
        tet_index_out(prefix_sum(TT(i, 3))++) = i;
    }
    //cout << "huuuuh " << prefix_sum << endl;
}

//compressed sparse row of vertices
//void TetMesh::csr(const Eigen::MatrixXd &V, const Eigen::MatrixXi &T, Eigen::VectorXi &prefix_sum, Eigen::VectorXi &idxs) {
//    using namespace std;
//    using namespace Eigen;
//}

void TetMesh::count_neighbors(const Eigen::MatrixXi TT, const Eigen::MatrixXi &AM, Eigen::VectorXd &out) {
    using namespace Eigen;

    VectorXi neighbors(AM.rows());
    for (unsigned i = 0; i < AM.rows(); ++i) {
        unsigned count = 0;
        for (unsigned j = 0; j < AM.cols(); ++j) {
            if (AM(i, j) != 0) {
                count++;
            }
        }
        neighbors(i) = count;
    }

    out.resize(TT.rows());
    for (unsigned i = 0; i < TT.rows(); ++i) {
        Vector4d res;
        res(0) = (double) neighbors(TT(i, 0));
        res(1) = (double) neighbors(TT(i, 1));
        res(2) = (double) neighbors(TT(i, 2));
        res(3) = (double) neighbors(TT(i, 3));
        out(i) = res.maxCoeff();
    }
}

inline double signed_volume(auto a, auto b, auto c, auto d) {
    using namespace Eigen;

    Matrix3d det;
    det << (a.x() - d.x()), (b.x() - d.x()), (c.x() - d.x())
            , (a.y() - d.y()), (b.y() - d.y()), (c.y() - d.y())
            , (a.z() - d.z()), (b.z() - d.z()), (c.z() - d.z());

    return det.determinant() * (1.0 / 6.0);
}

inline double max_dihedral_angle(auto a, auto b, auto c, auto d) {
    using namespace Eigen;
    using namespace std;

    Eigen::Matrix<double, Eigen::Dynamic, 3> normal(4, 3);
    normal.row(0) = (a - c).cross(d - c).normalized();
    normal.row(1) = (c - b).cross(d - b).normalized();
    normal.row(2) = (b - a).cross(d - a).normalized();
    normal.row(3) = (a - b).cross(c - b).normalized();

    VectorXd cartesian_normals(6);
    unsigned int k = 0;
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned m = n + 1; m < 4; ++m) {
            auto cos_angle = clamp(normal.row(n).dot(normal.row(m)), -1.0, 1.0);
            cartesian_normals(k) = M_PI - std::acos(cos_angle);
            k++;
        }
    }
    return cartesian_normals.maxCoeff();
}

inline double min_dihedral_angle(auto a, auto b, auto c, auto d) {
    using namespace Eigen;
    using namespace std;

    Eigen::Matrix<double, Eigen::Dynamic, 3> normal(4, 3);
    normal.row(0) = (a - c).cross(d - c).normalized();
    normal.row(1) = (c - b).cross(d - b).normalized();
    normal.row(2) = (b - a).cross(d - a).normalized();
    normal.row(3) = (a - b).cross(c - b).normalized();

    VectorXd cartesian_normals(6);
    unsigned int k = 0;
    for (unsigned n = 0; n < 4; ++n) {
        for (unsigned m = n + 1; m < 4; ++m) {
            auto cos_angle = clamp(normal.row(n).dot(normal.row(m)), -1.0, 1.0);
            cartesian_normals(k) = M_PI - std::acos(cos_angle);
            k++;
        }
    }
    return cartesian_normals.minCoeff();
}

// See "What is a good finite element" Jonathan Richard Shewchuk p. 61
void TetMesh::compute_volumes(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, Eigen::VectorXd &out) {
    using namespace Eigen;

    out.resize(TT.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        const auto V = signed_volume(a, b, c, d);
        //det.determinant() * (1.0/6.0);

        //const auto V =
        //    (a.x()-d.x())*(b.y()-d.y())*(c.z()-d.z())
        //  + (b.x()-d.x())*(c.y()-d.y())*(a.z()-d.z())
        //  + (c.x()-d.x())*(a.y()-d.y())*(b.z()-d.z())
        //  - (c.x()-d.x())*(b.y()-d.y())*(a.z()-d.z())
        //  - (b.x()-d.x())*(a.y()-d.y())*(c.z()-d.z())
        //  - (a.x()-d.x())*(c.y()-d.y())*(b.z()-d.z());

        out(i) = V;
    }
}


void TetMesh::compute_boundary_flags(const Eigen::MatrixXd &TV, const Eigen::MatrixXi &TF, Eigen::VectorXi &out) {
    out.setZero(TV.rows());

    for (unsigned i = 0; i < TF.rows(); ++i) {
        out(TF(i, 0)) = 1;
        out(TF(i, 1)) = 1;
        out(TF(i, 2)) = 1;
    }
}

void TetMesh::compute_boundary_flags(const Eigen::MatrixXd &TV, const Eigen::MatrixXi &TF, Eigen::VectorXd &out) {
    out.setZero(TV.rows());

    for (unsigned i = 0; i < TF.rows(); ++i) {
        out(TF(i, 0)) = 1.0;
        out(TF(i, 1)) = 1.0;
        out(TF(i, 2)) = 1.0;
    }
}

void TetMesh::compute_is_delaunay(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, const Eigen::MatrixXi &AM,
                                  Eigen::VectorXd &out) {
    using namespace Eigen;
    auto in_sphere = [&](auto a, auto b, auto c, auto d, auto p) {
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

        const double a_sqr = ax * ax + ay * ay + az * az;
        const double b_sqr = bx * bx + by * by + bz * bz;
        const double c_sqr = cx * cx + cy * cy + cz * cz;
        const double d_sqr = dx * dx + dy * dy + dz * dz;

        Matrix<double, 4, 4> det;
        det << ax, ay, az, a_sqr,
                bx, by, bz, b_sqr,
                cx, cy, cz, c_sqr,
                dx, dy, dz, d_sqr;

        return det.determinant();
    };
    out.resize(TT.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        auto flag = 1e-13;
        for (unsigned j = 0; j < TV.rows(); ++j) {
            const auto p = TV.row(j);
            if (j != TT(i, 0) &&
                j != TT(i, 1) &&
                j != TT(i, 2) &&
                j != TT(i, 3)) {
                auto res = in_sphere(a, b, c, d, p);
                if (res > 1e-14) {
                    flag = res;
                }
            }
        }
        out(i) = flag;
    }
}

// See "What is a good finite element" Jonathan Richard Shewchuk p. 54
void TetMesh::area_volume_ratio(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, const Eigen::VectorXd &volumes,
                                Eigen::VectorXd &out) {
    // This returns 4A^2
    auto area = [&](auto a, auto b, auto c) {
        auto yz = ((a.y() - c.y()) * (b.z() - c.z())) - ((b.y() - c.y()) * (a.z() - c.z()));
        auto zx = ((a.z() - c.z()) * (b.x() - c.x())) - ((b.z() - c.z()) * (a.x() - c.x()));
        auto xy = ((a.x() - c.x()) * (b.y() - c.y())) - ((b.x() - c.x()) * (a.y() - c.y()));
        return (yz * yz + zx * zx + xy * xy) * 0.25;
    };

    out.resize(TT.rows());
    for (unsigned i = 0; i < TT.rows(); ++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // tetgen uses a different vertex order than Shewchuk...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        double a1 = area(d, b, c);
        double a2 = area(a, d, c);
        double a3 = area(a, d, b);
        double a4 = area(a, b, c);

        double area_sum = pow(a1 + a2 + a3 + a4, 0.75);
        out(i) = (volumes(i) / area_sum) * 6.83852117086433292598068; //3^(7/4)
    }
}

void TetMesh::insphere_to_circumsphere(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV,
                                       const Eigen::VectorXd &volumes, Eigen::VectorXd &out) {
    using namespace Eigen;
    using namespace std;

    auto area = [&](auto a, auto b, auto c) {
        auto yz = ((a.y() - c.y()) * (b.z() - c.z())) - ((b.y() - c.y()) * (a.z() - c.z()));
        auto zx = ((a.z() - c.z()) * (b.x() - c.x())) - ((b.z() - c.z()) * (a.x() - c.x()));
        auto xy = ((a.x() - c.x()) * (b.y() - c.y())) - ((b.x() - c.x()) * (a.y() - c.y()));
        return std::sqrt(yz * yz + zx * zx + xy * xy) * 0.5;
    };

    out.resize(TT.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        const Vector3d t = (a - d);
        const Vector3d u = (b - d);
        const Vector3d v = (c - d);

        const double tabs = t.dot(t); //t dot t = |t|^2
        const double uabs = u.dot(u);
        const double vabs = v.dot(v);

        double Z = ((tabs * u).cross(v) + (uabs * v).cross(t) + (vabs * t).cross(u)).norm();

        double a1 = area(d, b, c);
        double a2 = area(a, d, c);
        double a3 = area(a, d, b);
        double a4 = area(a, b, c);
        double A = a1 + a2 + a3 + a4;

        out(i) = 108.0 * ((volumes(i) * volumes(i)) / (Z * A));
    }
}

void TetMesh::compute_aspect_ratios(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV,
                                    const Eigen::VectorXd &volumes, Eigen::VectorXd &out) {
    using namespace Eigen;
    using namespace std;

    out.resize(TT.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        const auto a = TV.row(TT(i, 0));
        const auto b = TV.row(TT(i, 2)); // weird vertex order...
        const auto c = TV.row(TT(i, 1));
        const auto d = TV.row(TT(i, 3));

        Eigen::Matrix<double, Eigen::Dynamic, 3> ls(6, 3);
        ls.row(0) = b - a;
        ls.row(1) = a - c;
        ls.row(2) = a - d;
        ls.row(3) = c - b;
        ls.row(4) = b - d;
        ls.row(5) = c - d;

        MatrixXd norms = ls.rowwise().norm();
        const double lmax = norms.maxCoeff();

        MatrixXd crosses = MatrixXd(15, 3);
        unsigned int k = 0;
        for (unsigned n = 0; n < 6; ++n) {
            for (unsigned m = n + 1; m < 6; ++m) {
                crosses.row(k) = ls.row(n).cross(ls.row(m));
                k++;
            }
        }
        auto max_crosswx = crosses.rowwise().norm().array().maxCoeff();

        out(i) = (volumes(i) / (max_crosswx * lmax)) * 6.0 * M_SQRT2;
    }
}

void TetMesh::compute_dihedral_angles(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, Eigen::VectorXd &out) {
    using namespace Eigen;
    using namespace std;

    out.resize(TT.rows());

    for (unsigned i = 0; i < TT.rows(); ++i) {
        const Vector3d a = TV.row(TT(i, 0));
        const Vector3d b = TV.row(TT(i, 2)); // weird vertex order...
        const Vector3d c = TV.row(TT(i, 1));
        const Vector3d d = TV.row(TT(i, 3));

        out(i) = min_dihedral_angle(a, b, c, d) / 1.230959417340774682134929178247; //arcsin(2sqrt(2)/3)
    }
}
void TetMesh::display(const Eigen::MatrixXi TT, const Eigen::MatrixXd TV,
                    Eigen::MatrixXi &dF,
                    Eigen::MatrixXd &dV) {
    dV.resize(TT.rows() * 4, 3);
    dF.resize(TT.rows() * 4, 3);

    for (unsigned i = 0; i < TT.rows(); ++i) {
        dV.row(i * 4 + 0) = TV.row(TT(i, 0));
        dV.row(i * 4 + 1) = TV.row(TT(i, 1));
        dV.row(i * 4 + 2) = TV.row(TT(i, 2));
        dV.row(i * 4 + 3) = TV.row(TT(i, 3));
        dF.row(i * 4 + 0) << (i * 4) + 0, (i * 4) + 1, (i * 4) + 3;
        dF.row(i * 4 + 1) << (i * 4) + 0, (i * 4) + 2, (i * 4) + 1;
        dF.row(i * 4 + 2) << (i * 4) + 3, (i * 4) + 2, (i * 4) + 0;
        dF.row(i * 4 + 3) << (i * 4) + 1, (i * 4) + 2, (i * 4) + 3;
    }
}

void TetMesh::slice(double slice_t, double filter_t, const Eigen::VectorXd _colors, Eigen::MatrixXi &dF,
                    Eigen::MatrixXd &dV, Eigen::MatrixXd &C) {
    using namespace std;
    using namespace Eigen;

    // Compute barycenters
    MatrixXd Bc;
    igl::barycenter(TV, TT, Bc);

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

    if (color.size() == TT.rows()) {
        for (int idx: sorted_i) {
            if (v(idx) <= slice_t && color(idx) >= filter_t) {
                tet_i.push_back(idx);
            }
        }
    } else {
        for (int idx: sorted_i) {
            if (v(idx) < slice_t) {
                tet_i.push_back(idx);
            }
        }
    }
    // make sure it's not empty
    if (tet_i.empty()) {
        tet_i.push_back(sorted_i[0]);
    }

    dV.resize(tet_i.size() * 4, 3);
    dF.resize(tet_i.size() * 4, 3);
    VectorXd dColors = VectorXd(dV.rows());
    C.resize(dV.rows(), 3);

    for (unsigned i = 0; i < tet_i.size(); ++i) {
        dV.row(i * 4 + 0) = TV.row(TT(tet_i[i], 0));
        dV.row(i * 4 + 1) = TV.row(TT(tet_i[i], 1));
        dV.row(i * 4 + 2) = TV.row(TT(tet_i[i], 2));
        dV.row(i * 4 + 3) = TV.row(TT(tet_i[i], 3));
        dF.row(i * 4 + 0) << (i * 4) + 0, (i * 4) + 1, (i * 4) + 3;
        dF.row(i * 4 + 1) << (i * 4) + 0, (i * 4) + 2, (i * 4) + 1;
        dF.row(i * 4 + 2) << (i * 4) + 3, (i * 4) + 2, (i * 4) + 0;
        dF.row(i * 4 + 3) << (i * 4) + 1, (i * 4) + 2, (i * 4) + 3;
    }

    if (color.size() == TT.rows()) {
        for (unsigned i = 0; i < tet_i.size(); ++i) {
            dColors(i * 4 + 0) = color(tet_i[i]);
            dColors(i * 4 + 1) = color(tet_i[i]);
            dColors(i * 4 + 2) = color(tet_i[i]);
            dColors(i * 4 + 3) = color(tet_i[i]);
        }
    } else {
        for (unsigned i = 0; i < tet_i.size(); ++i) {
            dColors(i * 4 + 0) = color(TT(tet_i[i], 0));
            dColors(i * 4 + 1) = color(TT(tet_i[i], 1));
            dColors(i * 4 + 2) = color(TT(tet_i[i], 2));
            dColors(i * 4 + 3) = color(TT(tet_i[i], 3));
        }
    }

    igl::jet(dColors, false, C);
}


void TetMesh::smooth(const double t) {
    using namespace std;
    using namespace Eigen;

    MatrixXd result = TV;

    for (unsigned k = 0; k < 25; ++k) {
        for (unsigned i = 0; i < result.rows(); ++i) {
            if (boundary_flag(i) == 1) continue;

            Vector3d p(0.0, 0.0, 0.0);
            unsigned count = 0;

            for (unsigned j = 0; j < result.rows(); ++j) {
                if (AM(i, j) == 1) {
                    // If adjacen
                    count++;
                }
            }

            double ratio = 1.0 / double(count);

            for (unsigned j = 0; j < result.rows(); ++j) {
                if (AM(i, j)) {
                    // If adjacent
                    p += (result.row(j) * ratio);
                }
            }
            p *= 1 - t;
            p += result.row(i) * t;

            //result.row(i) /= count;
            result.row(i) = p;
        }
    }

    TV = result;

    this->points_changed();
}

void TetMesh::spatial_sort(Eigen::MatrixXd& TV, Eigen::MatrixXi& TT) {
    using namespace std;
    using namespace Eigen;

    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef K::Point_3 Point_3;

    const int num_vertices = TV.rows();
    const int num_tets = TT.rows();

    // Step 1: Create a vector of pairs (point, original_index)
    typedef pair<Point_3, size_t> Point_with_index;
    vector<Point_with_index> points_with_indices;
    points_with_indices.reserve(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        points_with_indices.push_back(
            make_pair(Point_3(TV(i, 0), TV(i, 1), TV(i, 2)), i)
        );
    }

    // Step 2: Create a custom spatial sorting traits that works with pairs
    struct Point_with_index_traits {
        typedef Point_with_index Point_3;  // Changed from Point_2 to Point_3

        struct Less_x_3 {
            bool operator()(const Point_3& p, const Point_3& q) const {
                return p.first.x() < q.first.x();
            }
        };

        struct Less_y_3 {
            bool operator()(const Point_3& p, const Point_3& q) const {
                return p.first.y() < q.first.y();
            }
        };

        struct Less_z_3 {
            bool operator()(const Point_3& p, const Point_3& q) const {
                return p.first.z() < q.first.z();
            }
        };

        Less_x_3 less_x_3_object() const { return Less_x_3(); }
        Less_y_3 less_y_3_object() const { return Less_y_3(); }
        Less_z_3 less_z_3_object() const { return Less_z_3(); }
    };

    // Step 3: Sort points using CGAL's spatial_sort
    CGAL::spatial_sort(
        points_with_indices.begin(),
        points_with_indices.end(),
        Point_with_index_traits()
    );

    // Step 4: Create mapping from old index to new index
    vector<int> old_to_new(num_vertices);
    for (size_t new_idx = 0; new_idx < points_with_indices.size(); ++new_idx) {
        size_t old_idx = points_with_indices[new_idx].second;
        old_to_new[old_idx] = new_idx;
    }

    // Step 5: Reorder vertices based on sorted indices
    MatrixXd TV_reordered(num_vertices, 3);
    for (size_t new_idx = 0; new_idx < points_with_indices.size(); ++new_idx) {
        size_t old_idx = points_with_indices[new_idx].second;
        TV_reordered.row(new_idx) = TV.row(old_idx);
    }

    // Step 6: Update tetrahedra indices to use new vertex ordering
    MatrixXi TT_reordered = TT;
    for (int t = 0; t < num_tets; ++t) {
        for (int v = 0; v < 4; ++v) {
            TT_reordered(t, v) = old_to_new[TT(t, v)];
        }
    }

    // Step 7: Sort tetrahedra by their first vertex (for additional locality)
    // Create pairs of (first_vertex_index, tet_index)
    vector<pair<int, int>> tet_ordering;
    tet_ordering.reserve(num_tets);
    for (int t = 0; t < num_tets; ++t) {
        // Use the minimum vertex index as sorting key for better locality
        int min_vertex = min({TT_reordered(t, 0), TT_reordered(t, 1),
                             TT_reordered(t, 2), TT_reordered(t, 3)});
        tet_ordering.push_back({min_vertex, t});
    }

    // Sort tetrahedra by minimum vertex index
    sort(tet_ordering.begin(), tet_ordering.end());

    // Reorder tetrahedra based on sorted order
    MatrixXi TT_final(num_tets, 4);
    for (int new_t = 0; new_t < num_tets; ++new_t) {
        int old_t = tet_ordering[new_t].second;
        TT_final.row(new_t) = TT_reordered.row(old_t);
    }

    // Update the original matrices
    TV = TV_reordered;
    TT = TT_final;

    cout << "Mesh reordered for cache locality using CGAL spatial_sort" << endl;
    cout << "Vertices: " << num_vertices << ", Tetrahedra: " << num_tets << endl;
}

void TetMesh::randomise_order(Eigen::MatrixXd& TV, Eigen::MatrixXi& TT) {
        using namespace std;
    using namespace Eigen;

    const int num_vertices = TV.rows();
    const int num_tets = TT.rows();
    const int seed = 42;

    // Create random number generator
    mt19937 rng(seed);

    // Step 1: Create random permutation of vertex indices
    vector<int> vertex_permutation(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        vertex_permutation[i] = i;
    }
    shuffle(vertex_permutation.begin(), vertex_permutation.end(), rng);

    // Step 2: Create mapping from old index to new index
    vector<int> old_to_new(num_vertices);
    for (int new_idx = 0; new_idx < num_vertices; ++new_idx) {
        int old_idx = vertex_permutation[new_idx];
        old_to_new[old_idx] = new_idx;
    }

    // Step 3: Reorder vertices based on random permutation
    MatrixXd TV_reordered(num_vertices, 3);
    for (int new_idx = 0; new_idx < num_vertices; ++new_idx) {
        int old_idx = vertex_permutation[new_idx];
        TV_reordered.row(new_idx) = TV.row(old_idx);
    }

    // Step 4: Update tetrahedra indices to use new vertex ordering
    MatrixXi TT_reordered = TT;
    for (int t = 0; t < num_tets; ++t) {
        for (int v = 0; v < 4; ++v) {
            TT_reordered(t, v) = old_to_new[TT(t, v)];
        }
    }

    // Step 5: Create random permutation of tetrahedra
    vector<int> tet_permutation(num_tets);
    for (int i = 0; i < num_tets; ++i) {
        tet_permutation[i] = i;
    }
    shuffle(tet_permutation.begin(), tet_permutation.end(), rng);

    // Step 6: Reorder tetrahedra based on random permutation
    MatrixXi TT_final(num_tets, 4);
    for (int new_t = 0; new_t < num_tets; ++new_t) {
        int old_t = tet_permutation[new_t];
        TT_final.row(new_t) = TT_reordered.row(old_t);
    }

    // Update the original matrices
    TV = TV_reordered;
    TT = TT_final;

    cout << "Mesh reordered with random ordering (seed=" << seed << ")" << endl;
    cout << "Vertices: " << num_vertices << ", Tetrahedra: " << num_tets << endl;
}
void TetMesh::flip23(int i1, int i2, Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN, const Eigen::MatrixXd &TV) {
    using namespace std;
    using namespace Eigen;

    auto tet1 = TT.row(i1);
    auto tet2 = TT.row(i2);

    int shared[3];
    int shared_tet1[3];
    int shared_tet2[3];

    unsigned idx = 0;
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            auto p1 = tet1(i);
            auto p2 = tet2(j);
            if (p1 == p2) {
                shared[idx] = p1;
                shared_tet1[idx] = i;
                shared_tet2[idx] = j;
                idx++;
            }
        }
    }

    assert(idx == 3);

    // Find the apex vertices (the vertices not on the shared face)
    int apex1 = -1, apex2 = -1;
    for (int v: tet1) {
        bool flag = false;
        for (int i = 0; i < 3; ++i) {
            if (shared[i] == v)
                flag = true;
        }
        if (!flag) {
            apex1 = v;
            break;
        }
    }

    for (int v: tet2) {
        bool flag = false;
        for (int i = 0; i < 3; ++i) {
            if (shared[i] == v)
                flag = true;
        }
        if (!flag) {
            apex2 = v;
            break;
        }
    }

    assert(apex1 != -1 && apex2 != -1);

    Vector3d a = TV.row(shared[0]);
    Vector3d b = TV.row(shared[1]);
    Vector3d c = TV.row(shared[2]);
    Vector3d p = TV.row(apex1);

    Vector3d ab = b - a;
    Vector3d ac = c - a;
    Vector3d ap = p - a;

    auto orientation_test = ap.dot(ab.cross(ac));

    if (orientation_test < 0.0)
        swap(shared[1], shared[2]);

    //Vector4i tet1_r(shared[0],  shared[1], shared[2], apex1);
    //Vector4i tet2_r(shared[0],  shared[2], shared[1], apex2);

    Vector4i tet1_r(apex2, apex1, shared[0], shared[1]);
    Vector4i tet2_r(apex2, apex1, shared[1], shared[2]);
    Vector4i tet3_r(apex2, apex1, shared[2], shared[0]);

    double V_1 = signed_volume(
        TV.row(tet1_r(0)),
        TV.row(tet1_r(2)),
        TV.row(tet1_r(1)),
        TV.row(tet1_r(3))
    );
    double V_2 = signed_volume(
        TV.row(tet2_r(0)),
        TV.row(tet2_r(2)),
        TV.row(tet2_r(1)),
        TV.row(tet2_r(3))
    );

    double V_3 = signed_volume(
        TV.row(tet3_r(0)),
        TV.row(tet3_r(2)),
        TV.row(tet3_r(1)),
        TV.row(tet3_r(3))
    );

    if (V_1 <= 0.0 || V_2 <= 0.0 || V_3 <= 0.0) {
        //todo do not flip
    }

    cout << "tet 1: " << TT.row(i1) << endl;
    cout << "tet 1: " << tet1_r << "\n vol: " << V_1 << endl;
    cout << "tet 2: " << TT.row(i2) << endl;
    cout << "tet 2: " << tet2_r << "\n vol: " << V_2 << endl;
    cout << "tet 3: " << tet3_r << "\n vol: " << V_3 << endl;

    //TT.row(i1) = tet1_r;
    //TT.row(i2) = tet2_r;
}


void TetMesh::flip32(int i1, int i2, int i3, Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN, const Eigen::MatrixXd &TV) {
    using namespace std;
    using namespace Eigen;

    auto tet1 = TT.row(i1);
    auto tet2 = TT.row(i2);
    auto tet3 = TT.row(i3);

    int apexes[2];

    unsigned idx = 0;
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            for (unsigned k = 0; k < 4; ++k) {
                auto p1 = tet1(i);
                auto p2 = tet2(j);
                auto p3 = tet3(k);
                if (p1 == p2 && p2 == p3) {
                    apexes[idx] = p1;
                    idx++;
                }
            }
        }
    }

    assert(idx == 2);

    int triangle[3];
    idx = 0;
    for (int v: tet1) {
        if (apexes[0] != v && apexes[1] != v) {
            triangle[idx] = v;
            idx++;
        }
    }

    assert(idx == 2);

    for (int v: tet2) {
        if (apexes[0] != v && apexes[1] != v && triangle[0] != v && triangle[1] != v) {
            triangle[2] = v;
            break;
        }
    }

    Vector3d a = TV.row(triangle[0]);
    Vector3d b = TV.row(triangle[1]);
    Vector3d c = TV.row(triangle[2]);
    Vector3d p = TV.row(apexes[0]);

    Vector3d ab = b - a;
    Vector3d ac = c - a;
    Vector3d ap = p - a;

    auto orientation_test = ap.dot(ab.cross(ac));

    if (orientation_test < 0.0)
        swap(triangle[1], triangle[2]);

    Vector4i tet1_r(triangle[0],  triangle[1], triangle[2], apexes[0]);
    Vector4i tet2_r(triangle[0],  triangle[2], triangle[1], apexes[1]);

    //Vector4i tet1_r(apex2, apexes, shared[0], shared[1]);
    //Vector4i tet2_r(apex2, , shared[1], shared[2]);
    //Vector4i tet3_r(apex2, , shared[2], shared[0]);

    double V_1 = signed_volume(
        TV.row(tet1_r(0)),
        TV.row(tet1_r(2)),
        TV.row(tet1_r(1)),
        TV.row(tet1_r(3))
    );
    double V_2 = signed_volume(
        TV.row(tet2_r(0)),
        TV.row(tet2_r(2)),
        TV.row(tet2_r(1)),
        TV.row(tet2_r(3))
    );

    if (V_1 <= 0.0 || V_2 <= 0.0 ) {
        //todo do not flip
    }

    cout << "tet 1: " << TT.row(i1) << endl;
    cout << "tet 1: " << tet1_r << "\n vol: " << V_1 << endl;
    cout << "tet 2: " << TT.row(i2) << endl;
    cout << "tet 2: " << tet2_r << "\n vol: " << V_2 << endl;
    cout << "tet 3: " << TT.row(i3) << endl;

    TT.row(i1) = tet1_r;
    TT.row(i2) = tet2_r;
    TT.row(i3) = tet2_r;
}

void TetMesh::flip_everything(const Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN, const Eigen::MatrixXd &TV) {
    using namespace std;
    using namespace Eigen;

    MatrixXi flips23, flips32, TF23;
    TetMesh::flips(TT, TN, flips23, flips32, TF23);

    VectorXd results = VectorXd(flips23.rows());
    VectorXi neighbors[6];

    VectorXi tet_sum;
    tet_sum.setZero()(TT.rows());
    //tet_sum.setOnes(TT.rows());

    //for (unsigned i = 0; i < flips23.rows();  ++i ) {
    //    auto flip = flips23.row(i);
    //    double measure = measure(flip);
    //    double result = flip23(flip(0),flip(1), TT, TN, TV);

    //    results(i) = result;

    //    //syncthreads()

    //    bool to_flip = true;
    //    if (result <= measure)
    //        to_flip = false;
    //    for (unsigned j = 0; j < 6; ++j) {
    //        if (result <= neighbors[j](i)) {
    //            to_flip = false;
    //            break;
    //        }
    //    }

    //    if (to_flip) {
    //        // todo resize TT
    //        // write new tets into TT

    //        //todo size of TN changes

    //        //todo update TT and TN
    //    }
    //}
}

void TetMesh::connectivity(const unsigned int idx, Eigen::VectorXi TV_to_TT_offsets, Eigen::VectorXi TV_to_TT) {
    using namespace std;
    using namespace Eigen;

    unsigned max_index = idx + 1 < TV_to_TT_offsets.size() ? TV_to_TT_offsets(idx + 1) : TV_to_TT.size();

    cout << "i: " << TV_to_TT_offsets(idx) << endl;
    cout << "max: " << max_index << endl;
    for (unsigned i = TV_to_TT_offsets(idx); i < max_index; ++i) {
        TT(TV_to_TT(i), 0) = 5;
        TT(TV_to_TT(i), 1) = 7;
        TT(TV_to_TT(i), 2) = 4;
        TT(TV_to_TT(i), 3) = 8;
    }
    cout << "haha" << endl;

    this->points_changed();
}
