#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include "lbm_shared.hpp"
#include "lbm_kokkos_harmonic3D.hpp"

namespace lbm_kokkos {

	void LBM::output_error() {
    // create mirror view
    auto h_R = Kokkos::create_mirror_view(R);
    Kokkos::deep_copy(h_R, R);
    
    std::ofstream fout("RMSE.dat");
    for (int i = 0; i <= NN; i++) {
        fout << i << "      " << h_R(i) << std::endl;
    }
    fout.close();
}
void LBM::output_subtractionx1() // output
{
	std::ostringstream name;
	name<<"subtraction_x=0.2.dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<std::endl;
	
	for(j=0;j<=Ny;j++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1(20,j,z)-V(20,j,z)<<std::endl;
		}
	}
}
void LBM::output_subtractionx2() // output
{
	std::ostringstream name;
	name<<"subtraction_x=0.8.dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<std::endl;
	
	for(j=0;j<=Ny;j++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1(80,j,z)-V(80,j,z)<<std::endl;
		}
	}
}
void LBM::output_subtractiony1() // output
{
	std::ostringstream name;
	name<<"subtraction_y=0.2.dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1(i,20,z)-V(i,20,z)<<std::endl;
		}
	}
}
void LBM::output_subtractiony2() // output
{
	std::ostringstream name;
	name<<"subtraction_y=0.8.dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",K="<<Np+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1(i,80,z)-V(i,80,z)<<std::endl;
		}
	}
}
void LBM::output_subtractionz1() // output
{
	std::ostringstream name;
	name<<"subtraction_z=0.2.dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",J="<<Np+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(j=0;j<=Ny;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1(i,j,20)-V(i,j,20)<<std::endl;
		}
	}
}
void LBM::output_subtractionz2() // output
{
	std::ostringstream name;
	name<<"subtraction_z=0.8.dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Np+1<<",J="<<Np+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(j=0;j<=Ny;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1(i,j,80)-V(i,j,80)<<std::endl;
		}
	}
}

void LBM::init()
{
	std::cout<<"tau_f="<<tau_f<<std::endl;
	std::cout<<"Lamda="<<Lamda<<std::endl;

	Precision w_init[Q]={1.0/2,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24};
	Precision w_bar_init[Q]={0,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12};
    Kokkos::parallel_for("w_init", Q,
                       [&](const int i) { w(i) = w_init[i]; });
    Kokkos::parallel_for("w_bar_init", Q,
                       [&](const int i) { w_bar(i) = w_bar_init[i]; });
	Kokkos::parallel_for(
	Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Q, 3}),
	KOKKOS_LAMBDA(const int i, const int j) {
		constexpr int table[13][3] = {
		{0,0,0},{1,1,0},{1,-1,0},{1,0,1},{1,0,-1},{0,1,1},{0,1,-1},
		{-1,-1,0},{-1,1,0},{-1,0,-1},{-1,0,1},{0,-1,-1},{0,-1,1}
		};
		e(i, j) = table[i][j];
	});
	for(i=0;i<=Nx;i++)
	{
		for(j=0;j<=Ny;j++)
		{
			for(z=0;z<=Nz;z++)
			{
				U(i,j,z)=0;//initiating the entire field with 0s
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq(i,j,z,k)=(1.0-w(k))*U(i,j,z);//calculating equilirum disribution fucntion
						f(i,j,z,k)=feq(i,j,z,k);//let equilirum disribution fucntion be the distribution fucntion for the initial step
					}
					else 
					{
						feq(i,j,z,k)=w(k)*U(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
				}
			}
		}
	}	
}
void LBM::init1()
{
	std::cout<<"tau_f="<<tau_f<<std::endl;
	std::cout<<"Lamda="<<Lamda<<std::endl;

	for(i=0;i<=Nx;i++)
	{
		for(j=0;j<=Ny;j++)
		{
			for(z=0;z<=Nz;z++)
			{
				U1(i,j,z)=0;
				for(k=0;k<Q;k++)
				{
					if(k==0)
					{
						feq(i,j,z,k)=(1.0-w(k))*U1(i,j,z) ;
						f(i,j,z,k)=feq(i,j,z,k);
					}
					else 
					{
						feq(i,j,z,k)=w(k)*U1(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
				}
			}
		}
	}	
}
void LBM::evolution()
{
	Kokkos::parallel_for("initialize",Kokkos::TeamPolicy<>((Nx+1)*(Ny+1),Kokkos::AUTO,Kokkos::AUTO),//calculating source term and equlibirum distribution fucntion for the preparation of evolution
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx+1);
		const int j = team.league_rank()%(Nx+1);
		
			Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
				g(i,j,z)=source*(9.0*pi*pi*pi*pi*cos(pi*double(i)/Np)*sin(pi*double(j)/Np)*sin(pi*double(z)/Np));
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q),[=](int k)
				{
					if(k==0)
					{
						feq(i,j,z,k)=(1.0-w(k))*U(i,j,z) ;
						f(i,j,z,k)=feq(i,j,z,k);
					}
					else 
					{
						feq(i,j,z,k)=w(k)*U(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
				});
			});
		
	});
	Kokkos::parallel_for("collision",Kokkos::TeamPolicy<>((Nx+1)*(Ny+1),Kokkos::AUTO,Kokkos::AUTO),//collision
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx+1);
		const int j = team.league_rank()%(Nx+1);
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					M(i,j,z,k)=f(i,j,z,k)+(feq(i,j,z,k)-f(i,j,z,k))/tau_f+w_bar(k)*(0.5-tau_f)*g(i,j,z)*Lamda/(Np*Np);
				});
			});
	});
	Kokkos::parallel_for("streaming",Kokkos::TeamPolicy<>((Nx-1)*(Ny-1),Kokkos::AUTO,Kokkos::AUTO),//streaming
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx-1) + 1;
		const int j = team.league_rank()%(Nx-1) + 1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1 , Nz  ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					int ip,jp,zp;
					ip=i-e(k,0);
					jp=j-e(k,1);
					zp=z-e(k,2);
					F(i,j,z,k)=M(ip,jp,zp,k);
				});
			});
	});
	
	Kokkos::parallel_for("calculation of Marcroscopic variables",Kokkos::TeamPolicy<>((Nx-1)*(Ny-1),Kokkos::AUTO,Kokkos::AUTO),//calculation of Marcroscopic variables(solutions of the target PDEs)
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx-1)+1;
		const int j = team.league_rank()%(Nx-1)+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, Nz ), [=](int z)
			{
				U0(i,j,z)=U(i,j,z);
				U(i,j,z)=0;
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					f(i,j,z,k)=F(i,j,z,k);
					U(i,j,z)+=f(i,j,z,k);
				});
			});
	});

	Kokkos::parallel_for("calculation of feq ready for the boundary conditions",Kokkos::TeamPolicy<>((Nx-1)*(Ny-1),Kokkos::AUTO,Kokkos::AUTO),//calculation of feq ready for the boundary conditions
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx-1)+1;
		const int j = team.league_rank()%(Nx-1)+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, Nz ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
					{
						feq(i,j,z,k)=(1.0-w(k))*U(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
					else 
					{
						feq(i,j,z,k)=w(k)*U(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
				});
			});
	});

////////////////////////////boundary conditions////////////////////////

	Kokkos::parallel_for("left and right walls",Kokkos::TeamPolicy<>(Ny+1,Kokkos::AUTO,Kokkos::AUTO),//left and right walls
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int j = team.league_rank();
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{	
			//  U(0,j,z)=-3*pi*pi*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
			//  U(Nx,j,z)=3*pi*pi*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
				U(0,j,z)=U(1,j,z);    //Neumann boundary conditions
				U(Nx,j,z)=U(Nx-1,j,z);//Neumann boundary conditions
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
				{
					feq(0,j,z,k)=(1.0-w(k))*U(0,j,z);
					feq(Nx,j,z,k)=(1.0-w(k))*U(Nx,j,z);
					f(0,j,z,k)=feq(0,j,z,k)+f(1,j,z,k)-feq(1,j,z,k);
					f(Nx,j,z,k)=feq(Nx,j,z,k)+f(Nx-1,j,z,k)-feq(Nx-1,j,z,k);
				}
				else 
				{
					feq(0,j,z,k)=w(k)*U(0,j,z);
					feq(Nx,j,z,k)=w(k)*U(Nx,j,z);
					f(0,j,z,k)=feq(0,j,z,k)+f(1,j,z,k)-feq(1,j,z,k);
					f(Nx,j,z,k)=feq(Nx,j,z,k)+f(Nx-1,j,z,k)-feq(Nx-1,j,z,k);
				}
				});
			});
	});

		Kokkos::parallel_for("front and back",Kokkos::TeamPolicy<>(Nx-1,Kokkos::AUTO,Kokkos::AUTO),//front and back walls
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
			//	U(i,Ny,z)=0;//Dirichlet boundary conditions
			//	U(i,0,z)=0; //Dirichlet boundary conditions
				U(i,Ny,z)=U(i,Ny-1,z);//Neumann boundary conditions
				U(i,0,z)=U(i,1,z);	  //Neumann boundary conditions
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
				{
					feq(i,0,z,k)=(1.0-w(k))*U(i,0,z);
					feq(i,Ny,z,k)=(1.0-w(k))*U(i,Ny,z);
					f(i,0,z,k)=feq(i,0,z,k)+f(i,1,z,k)-feq(i,1,z,k);
					f(i,Ny,z,k)=feq(i,Ny,z,k)+f(i,Ny-1,z,k)-feq(i,Ny-1,z,k);
				}
				else
				{
					feq(i,0,z,k)=w(k)*U(i,0,z);
					feq(i,Ny,z,k)=w(k)*U(i,Ny,z);
					f(i,0,z,k)=feq(i,0,z,k)+f(i,1,z,k)-feq(i,1,z,k);
					f(i,Ny,z,k)=feq(i,Ny,z,k)+f(i,Ny-1,z,k)-feq(i,Ny-1,z,k);
				}
				});
			});
	});

		Kokkos::parallel_for("up and down",Kokkos::TeamPolicy<>(Nx-1,Kokkos::AUTO,Kokkos::AUTO),//up and down walls
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, Ny ), [=](int j)
			{
			//	U(i,j,0)=0; //Dirichlet boundary conditions
			//	U(i,j,Nz)=0;//Dirichlet boundary conditions
				U(i,j,0)=U(i,j,1);    //Neumann boundary conditions
				U(i,j,Nz)=U(i,j,Nz-1);//Neumann boundary conditions
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
				{
					feq(i,j,0,k)=(1.0-w(k))*U(i,j,0);
					feq(i,j,Nz,k)=(1.0-w(k))*U(i,j,Nz);
					f(i,j,0,k)=feq(i,j,0,k)+f(i,j,1,k)-feq(i,j,1,k);
					f(i,j,Nz,k)=feq(i,j,Nz,k)+f(i,j,Nz-1,k)-feq(i,j,Nz-1,k);
				}
				else 
				{
					feq(i,j,0,k)=w(k)*U(i,j,0);
					feq(i,j,Nz,k)=w(k)*U(i,j,Nz);
					f(i,j,0,k)=feq(i,j,0,k)+f(i,j,1,k)-feq(i,j,1,k);
					f(i,j,Nz,k)=feq(i,j,Nz,k)+f(i,j,Nz-1,k)-feq(i,j,Nz-1,k);
				}
				});
			});
	});

////////////////////////////done////////////////////////////////
}
void LBM::evolution1()
{
	Kokkos::parallel_for("initialize",Kokkos::TeamPolicy<>((Nx+1)*(Ny+1),Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx+1);
		const int j = team.league_rank()%(Nx+1);
		
			Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q),[=](int k)
				{
					if(k==0)
					{
						feq(i,j,z,k)=(1.0-w(k))*U1(i,j,z) ;
						f(i,j,z,k)=feq(i,j,z,k);
					}
					else 
					{
						feq(i,j,z,k)=w(k)*U1(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
				});
			});
		
	});
	Kokkos::parallel_for("collision",Kokkos::TeamPolicy<>((Nx+1)*(Ny+1),Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx+1);
		const int j = team.league_rank()%(Nx+1);
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					M(i,j,z,k)=f(i,j,z,k)+(feq(i,j,z,k)-f(i,j,z,k))/tau_f+w_bar(k)*(0.5-tau_f)*U(i,j,z)*Lamda/(Np*Np);
				});
			});
	});
	Kokkos::parallel_for("streaming",Kokkos::TeamPolicy<>((Nx-1)*(Ny-1),Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx-1)+1;
		const int j = team.league_rank()%(Nx-1)+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1 , Nz ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{	
					int ip,jp,zp;
					ip=i-e(k,0);
					jp=j-e(k,1);
					zp=z-e(k,2);
					F(i,j,z,k)=M(ip,jp,zp,k);


				});
			});
	});
	
	Kokkos::parallel_for("calculation of Marcroscopic variables",Kokkos::TeamPolicy<>((Nx-1)*(Ny-1),Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx-1)+1;
		const int j = team.league_rank()%(Nx-1)+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, Nz ), [=](int z)
			{
				U0(i,j,z)=U1(i,j,z);
				U1(i,j,z)=0;
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					f(i,j,z,k)=F(i,j,z,k);
					U1(i,j,z)+=f(i,j,z,k);
				});
			});
	});

	Kokkos::parallel_for("calculation of feq ready for the boundary conditions",Kokkos::TeamPolicy<>((Nx-1)*(Ny-1),Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()/(Nx-1)+1;
		const int j = team.league_rank()%(Nx-1)+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, Nz ), [=](int z)
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
					{
						feq(i,j,z,k)=(1.0-w(k))*U1(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
					else 
					{
						feq(i,j,z,k)=w(k)*U1(i,j,z);
						f(i,j,z,k)=feq(i,j,z,k);
					}
				});
			});
	});

////////////////////////////boundary conditions////////////////////////

	Kokkos::parallel_for("left and right walls",Kokkos::TeamPolicy<>(Ny+1,Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int j = team.league_rank();
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
				U1(0,j,z)=sin(pi*double(j)/Np)*sin(pi*double(z)/Np);  //Dirichlet boundary conditions
				U1(Nx,j,z)=-sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
				{
					feq(0,j,z,k)=(1.0-w(k))*U1(0,j,z);
					feq(Nx,j,z,k)=(1.0-w(k))*U1(Nx,j,z);
					f(0,j,z,k)=feq(0,j,z,k)+f(1,j,z,k)-feq(1,j,z,k);
					f(Nx,j,z,k)=feq(Nx,j,z,k)+f(Nx-1,j,z,k)-feq(Nx-1,j,z,k);
				}
				else 
				{
					feq(0,j,z,k)=w(k)*U1(0,j,z);
					feq(Nx,j,z,k)=w(k)*U1(Nx,j,z);
					f(0,j,z,k)=feq(0,j,z,k)+f(1,j,z,k)-feq(1,j,z,k);
					f(Nx,j,z,k)=feq(Nx,j,z,k)+f(Nx-1,j,z,k)-feq(Nx-1,j,z,k);
				}
				});
			});
	});

		Kokkos::parallel_for("front and back",Kokkos::TeamPolicy<>(Nx-1,Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,  Nz + 1 ), [=](int z)
			{
				U1(i,Ny,z)=0;//Dirichlet boundary conditions
				U1(i,0,z)=0; //Dirichlet boundary conditions
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
				{
					feq(i,0,z,k)=(1.0-w(k))*U1(i,0,z);
					feq(i,Ny,z,k)=(1.0-w(k))*U1(i,Ny,z);
					f(i,0,z,k)=feq(i,0,z,k)+f(i,1,z,k)-feq(i,1,z,k);
					f(i,Ny,z,k)=feq(i,Ny,z,k)+f(i,Ny-1,z,k)-feq(i,Ny-1,z,k);
				}
				else
				{
					feq(i,0,z,k)=w(k)*U1(i,0,z);
					feq(i,Ny,z,k)=w(k)*U1(i,Ny,z);
					f(i,0,z,k)=feq(i,0,z,k)+f(i,1,z,k)-feq(i,1,z,k);
					f(i,Ny,z,k)=feq(i,Ny,z,k)+f(i,Ny-1,z,k)-feq(i,Ny-1,z,k);
				}
				});
			});
	});

		Kokkos::parallel_for("up and down",Kokkos::TeamPolicy<>(Nx-1,Kokkos::AUTO,Kokkos::AUTO),
	KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type&team)
	{
		const int i = team.league_rank()+1;
		Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, Ny ), [=](int j)
			{
				U1(i,j,0)=0; //Dirichlet boundary conditions
				U1(i,j,Nz)=0;//Dirichlet boundary conditions
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,Q), [=](int k)
				{
					if(k==0)
				{
					feq(i,j,0,k)=(1.0-w(k))*U1(i,j,0);
					feq(i,j,Nz,k)=(1.0-w(k))*U1(i,j,Nz);
					f(i,j,0,k)=feq(i,j,0,k)+f(i,j,1,k)-feq(i,j,1,k);
					f(i,j,Nz,k)=feq(i,j,Nz,k)+f(i,j,Nz-1,k)-feq(i,j,Nz-1,k);
				}
				else 
				{
					feq(i,j,0,k)=w(k)*U1(i,j,0);
					feq(i,j,Nz,k)=w(k)*U1(i,j,Nz);
					f(i,j,0,k)=feq(i,j,0,k)+f(i,j,1,k)-feq(i,j,1,k);
					f(i,j,Nz,k)=feq(i,j,Nz,k)+f(i,j,Nz-1,k)-feq(i,j,Nz-1,k);
				}
				});
			});
	});

////////////////////////////done////////////////////////////////
}
void LBM::output_x1(int m) // output
{
	std::ostringstream name;
	name<<"3D-Laplace_yz_x=0.2-plane_"<<m<<".dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<Nx+1<<",K="<<Ny+1<<",F=POINT"<<std::endl;
	
	for(j=0;j<=Ny;j++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1(20,j,z)<<std::endl;
		}
	}
}
void LBM::output_x2(int m) // output
{
	std::ostringstream name;
	name<<"3D-Laplace_yz_x=0.8-plane_"<<m<<".dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"Y\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",J="
	<<Nx+1<<",K="<<Ny+1<<",F=POINT"<<std::endl;
	
	for(j=0;j<=Ny;j++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<U1(80,j,z)<<std::endl;
		}
	}
}
void LBM::output_y1(int m) // output
{
	std::ostringstream name;
	name<<"3D-Laplace_xz_y=0.2-plane_"<<m<<".dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Nx+1<<",K="<<Nz+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1(i,20,z)<<std::endl;
		}
	}
}
void LBM::output_y2(int m) // output
{
	std::ostringstream name;
	name<<"3D-Laplace_xz_y=0.8-plane_"<<m<<".dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Z\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Nx+1<<",K="<<Nz+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(z=0;z<=Nz;z++)
		{
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<U1(i,80,z)<<std::endl;
		}
	}
}
void LBM::output_z1(int m) // output
{
	std::ostringstream name;
	name<<"3D-Laplace_xy_z=0.2-plane_"<<m<<".dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Nx+1<<",J="<<Ny+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(j=0;j<=Ny;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1(i,j,20)<<std::endl;
		}
	}
}
void LBM::output_z2(int m) // output
{
	std::ostringstream name;
	name<<"3D-Laplace_xy_z=0.8-plane_"<<m<<".dat";
	std::ofstream out(name.str().c_str());
	out<<"Title=\"Laplace equation\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\"\n"<<"ZONE T=\"BOX\",I="
	<<Nx+1<<",J="<<Ny+1<<",F=POINT"<<std::endl;
	
	for(i=0;i<=Nx;i++)
	{
		for(j=0;j<=Ny;j++)
		{
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<U1(i,j,80)<<std::endl;
		}
	}
}
void LBM::Error()
{
	double temp1,temp2;
	temp1=0;
	temp2=0;
	for(i=1;i<Nx;i++)
	{
		for(j=1;j<Ny;j++)
		{
			for(z=1;z<Nz;z++)
			{
				temp1+=(U(i,j,z)-U0(i,j,z))*(U(i,j,z)-U0(i,j,z));
				temp2+=(U(i,j,z)*U(i,j,z));
			}
		}
	}
	temp1=sqrt(temp1);
	temp2=sqrt(temp2);
	error(0) = static_cast<Precision>(temp1/(temp2+1e-30));
}
void LBM::Error1()
{
	double temp1,temp2;
	temp1=0;
	temp2=0;
	for(i=1;i<Nx;i++)
	{
		for(j=1;j<Ny;j++)
		{
			for(z=1;z<Nz;z++)
			{
				temp1+=(U1(i,j,z)-U0(i,j,z))*(U1(i,j,z)-U0(i,j,z));
				temp2+=(U1(i,j,z)*U1(i,j,z));
			}
		}
	}
	temp1=sqrt(temp1);
	temp2=sqrt(temp2);
	error(0) = static_cast<Precision>(temp1/(temp2+1e-30));
}
}
