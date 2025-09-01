#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/barycenter.h>
#include <igl/stb/read_image.h>
#include <iostream>

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

// matcap texture
Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;

bool displayTetrahedralMesh = true;


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

  viewer.data().clear();
  viewer.data().set_mesh(dV,dF);
  viewer.data().set_face_based(true);
  viewer.data().set_texture(R,G,B,A);
  viewer.data().use_matcap = true;
}


void change_mesh(igl::opengl::glfw::Viewer& viewer) 
{
    displayTetrahedralMesh = !displayTetrahedralMesh;
    viewer.data().clear();

    if(displayTetrahedralMesh)
    {
      viewer.data().set_mesh(dV, dF);
    }
    else 
    {
      viewer.data().set_mesh(V,F);
    }
    viewer.data().set_texture(R,G,B,A);
    viewer.data().use_matcap = true;
    viewer.data().set_face_based(true);
}


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  std::cout << "Input: " << "0x" << std::hex << static_cast<unsigned int>(static_cast<unsigned char>(key)) << std::endl;

  if (key == 0x20) {
    change_mesh(viewer);
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

  // Add matcap
  igl::stb::read_image("../matcap/ceramic_dark.png", R,G,B,A); 

  
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.core().background_color.setConstant(0.3f);
  viewer.callback_key_down = &key_down;
  key_down(viewer, '5', 0);
  viewer.launch();
}
