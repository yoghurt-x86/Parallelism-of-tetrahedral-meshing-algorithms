#pragma once

#include <Eigen/Dense>
#include <vector>

class TetMesh {
private:
public:
    void points_changed();
    static void adjacency(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::MatrixXi &AM);
    static void csr_from_AM(const Eigen::MatrixXi &AM, Eigen::VectorXi &prefix_sum, Eigen::VectorXi &idxs);
    static void compute_aspect_ratios(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out);
    static void compute_volumes(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out);
    static void area_volume_ratio(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out);
    static void insphere_to_circumsphere(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::VectorXd &volumes, Eigen::VectorXd &out);
    static void compute_dihedral_angles(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, Eigen::VectorXd &out);
    static void count_neighbors(const Eigen::MatrixXi TT, const Eigen::MatrixXi& AM, Eigen::VectorXd &out);
    static void compute_is_delaunay(const Eigen::MatrixXi& TT,  const Eigen::MatrixXd& TV, const Eigen::MatrixXi &AM, Eigen::VectorXd &out);
    static void edge_pairs_from_TT(const Eigen::MatrixXi &TT, Eigen::MatrixXi &edges);
    static void compute_boundary_flags(const Eigen::MatrixXd &TV, const Eigen::MatrixXi &TF, Eigen::VectorXi &out);
    static void compute_boundary_flags(const Eigen::MatrixXd &TV, const Eigen::MatrixXi &TF, Eigen::VectorXd &out);
    static void vertex_to_TT_map(const Eigen::MatrixXi &TT, const Eigen::MatrixXd &TV, Eigen::VectorXi &offset_out, Eigen::VectorXi &tet_index_out);
    static void flips(const Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN,
                        Eigen::MatrixXi &two_three_flips,
                        Eigen::MatrixXi &three_two_flips,
                        Eigen::MatrixXi &TF23
                        );
    static void normalize_mesh(const Eigen::MatrixXd& V, Eigen::MatrixXd& V_out);
    static void flip23(int i1, int i2, Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN, const Eigen::MatrixXd &TV);
    static void flip32(int i1, int i2, int i3, Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN, const Eigen::MatrixXd &TV);
    static void flip_everything(const Eigen::MatrixXi &TT, const Eigen::MatrixXi &TN, const Eigen::MatrixXd &TV);
    static void spatial_sort(Eigen::MatrixXd& TV, Eigen::MatrixXi& TT);
    static void randomise_order(Eigen::MatrixXd& TV, Eigen::MatrixXi& TT);
    static void display(const Eigen::MatrixXi TT, const Eigen::MatrixXd TV, Eigen::MatrixXi &dF, Eigen::MatrixXd &dV);
    // Data
  //
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TF;
    Eigen::MatrixXi AM;
    Eigen::VectorXd volumes;
    Eigen::VectorXd av_ratio;
    Eigen::VectorXd in_circum_ratio;
    Eigen::VectorXd aspect_ratios;
    Eigen::VectorXd dihedral_angles;
    Eigen::VectorXd max_vertex_neigbors;
    Eigen::VectorXd is_delaunay;
    Eigen::VectorXi boundary_flag;
    Eigen::VectorXd boundary_flag_d;

    TetMesh();
    TetMesh(const Eigen::MatrixXd& TV, const Eigen::MatrixXi& TT, const Eigen::MatrixXi& TF);
    void slice(double slice_t, double ratio_t, const Eigen::VectorXd _colors, Eigen::MatrixXi &dF, Eigen::MatrixXd &dV, Eigen::MatrixXd &C);
    void smooth(const double t);
    void connectivity(const unsigned int i, Eigen::VectorXi offsets, Eigen::VectorXi TV_to_TT);
};
