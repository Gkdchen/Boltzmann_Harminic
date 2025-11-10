#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "lbm_kokkos_harmonic3D_impl.cuh"
#include <iostream>

namespace lbm_kokkos {
    void LBM::datainit(const LBMParams& p)
{
    Nx=p.Nx;
    Ny=p.Ny;
    Nz=p.Nz;
    Np=p.Np;
    source=p.source;
    Lamda=p.lamda;
    tau_f=p.tau_f;
    save_iter=p.save_iter;
}
    void LBM::allocate_views()
{
    U      = DataArray3D("U", Nx+1, Ny+1, Nz+1);
    V      = DataArray3D("V", Nx+1, Ny+1, Nz+1);
    U0     = DataArray3D("U0", Nx+1, Ny+1, Nz+1);
    U1     = DataArray3D("U1", Nx+1, Ny+1, Nz+1);
    f      = DataArray4D("f", Nx+1, Ny+1, Nz+1, Q);
    F      = DataArray4D("F", Nx+1, Ny+1, Nz+1, Q);
    M      = DataArray4D("M", Nx+1, Ny+1, Nz+1, Q);
    feq    = DataArray4D("feq", Nx+1, Ny+1, Nz+1, Q);
    g      = DataArray3D("g", Nx+1, Ny+1, Nz+1);
    w      = DataVectorD("w", Q);
    w_bar  = DataVectorD("w_bar", Q);
    R      = DataVectorD("R",save_iter);
    e      = DataArray2I("e", Q, 3);
    err    = DataValue0D("err");
    temp1  = DataValue0D("temp1");
    temp2  = DataValue0D("temp2");
    RSME   = DataValue0D("Rsme");
    Kokkos::fence();
    h_U    = Kokkos::create_mirror_view(U);
    h_V    = Kokkos::create_mirror_view(V);
    h_U0   = Kokkos::create_mirror_view(U0);
    h_U1   = Kokkos::create_mirror_view(U1);
    h_R    = Kokkos::create_mirror_view(R);
    h_e    = Kokkos::create_mirror_view(e);
    h_w    = Kokkos::create_mirror_view(w);
    h_w_bar= Kokkos::create_mirror_view(w_bar);
    h_temp1= Kokkos::create_mirror_view(temp1);
    h_temp2= Kokkos::create_mirror_view(temp2);
    h_err  = Kokkos::create_mirror_view(err);
    h_RSME = Kokkos::create_mirror_view(RSME);
}

	void LBM::output_error() {

    std::ofstream fout("RMSE.dat");
    for (int i = 0; i <= NN; i++) {
        fout << i << "      " << h_R(i) << std::endl;
    }
    fout.close();
}


void LBM::output_init() // output
{
/////////copy device view to host//////////
	Kokkos::deep_copy(h_U, U);
	Kokkos::deep_copy(h_U1, U1);
	Kokkos::deep_copy(h_V, V);
	Kokkos::fence();
///////////////////////////////////////////
}

void LBM::rsme()
{
///////create ptr of device view//////
    auto d_U1 = U1;
    auto d_V  = V;
    auto d_RSME = RSME;
//////////////////////////////////////
    Kokkos::deep_copy(RSME, 0.0);//init device viewpoint
	// caculate RSME
	Kokkos::parallel_reduce("compute_RSME",
		Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {Nx+1, Ny+1, Nz+1}),
		KOKKOS_LAMBDA (int i, int j, int z, double& local_RSME) {
			double diff = d_U1(i, j, z) - d_V(i, j, z);
			local_RSME += diff * diff;
		}, Kokkos::Sum<Precision,Device>(d_RSME));
    Kokkos::deep_copy(h_RSME,d_RSME);
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
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<h_U1(20,j,z)-h_V(20,j,z)<<std::endl;
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
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<h_U1(80,j,z)-h_V(80,j,z)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<h_U1(i,20,z)-h_V(i,20,z)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<h_U1(i,80,z)-h_V(i,80,z)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<h_U1(i,j,20)-h_V(i,j,20)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<h_U1(i,j,80)-h_V(i,j,80)<<std::endl;
		}
	}
}

void LBM::init()
{
	std::cout<<"tau_f="<<tau_f<<std::endl;
	std::cout<<"Lamda="<<Lamda<<std::endl;
	std::vector<double> w_init;       
	std::vector<double> w_bar_init;
	std::vector<std::vector<int>> e_init;   
	w_init={1.0/2,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24,1.0/24};
	w_bar_init={0,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12,1.0/12};
	e_init = {
		{0,0,0},{1,1,0},{1,-1,0},{1,0,1},{1,0,-1},{0,1,1},{0,1,-1},
		{-1,-1,0},{-1,1,0},{-1,0,-1},{-1,0,1},{0,-1,-1},{0,-1,1}
		};
    for(int i=0;i<Q;i++)
    { h_w(i) = w_init[i]; }
    for(int i=0;i<Q;i++)
    { h_w_bar(i) = w_bar_init[i]; }
	for(int i=0;i<Q;i++)
    {
	for(int j=0;j<3;j++)
    { h_e(i, j) = e_init[i][j]; }}
    Kokkos::fence();
    Kokkos::deep_copy(w,h_w);
    Kokkos::deep_copy(w_bar,h_w_bar);
    Kokkos::deep_copy(e,h_e);
///////allocate device ptr//////
    auto d_U   = U;
    auto d_f   = f;
    auto d_feq = feq;
    auto d_w   = w;
    int  d_Ny  = Ny;
    int  d_Nz  = Nz;
    int  d_Q   = Q;
/////////////////////////////////
    Kokkos::parallel_for("init",
        Kokkos::TeamPolicy<ExecSpace>((Nx+1)*(Ny+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
        {
            const int gid = team.league_rank();
            const int i   = gid / (d_Ny + 1);
            const int j   = gid % (d_Ny + 1);

            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
                [=](int z)
                {
                    d_U(i, j, z) = 0.0;//initiating the entire field with 0s
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                        [=](int k)
                        {
                            if (k < 0 || k >= d_Q) return;
                            double weight = d_w(k);
                            if(k==0)
                        {
                            d_feq(i, j, z, k) = (1-weight) * d_U(i, j, z);//calculating equilirum disribution fucntion
                            d_f(i, j, z, k)   = d_feq(i, j, z, k);//let equilirum disribution fucntion be the distribution fucntion for the initial step
                        }
                            else 
                        {
                            d_feq(i,j,z,k)=weight*d_U(i,j,z);
                            d_f(i,j,z,k)=d_feq(i,j,z,k);
                        }
                        });
                });
        });
}
void LBM::init1()
{
	std::cout<<"tau_f="<<tau_f<<std::endl;
	std::cout<<"Lamda="<<Lamda<<std::endl;
    auto d_U1  = U1;
    auto d_f   = f;
    auto d_feq = feq;
    auto d_w   = w;
    int  d_Ny  = Ny;
    int  d_Nz  = Nz;
    int  d_Q   = Q;
    Kokkos::parallel_for("init1",
    Kokkos::TeamPolicy<ExecSpace>((Nx+1)*(Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny + 1);
        const int j   = gid % (d_Ny + 1);

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
                d_U1(i, j, z) = 0.0;
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        double weight = d_w(k);
                        if(k==0)
					{
                        d_feq(i, j, z, k) = (1-weight) * d_U1(i, j, z);
                        d_f(i, j, z, k)   = d_feq(i, j, z, k);
                    }
					    else 
					{
                        d_feq(i,j,z,k)=weight*d_U1(i,j,z);
						d_f(i,j,z,k)=d_feq(i,j,z,k);
                    }
                    });
            });
    });
}
void LBM::evolution()
{
    auto d_U   = U;
    auto d_U0  = U0;
    auto d_f   = f;
    auto d_F   = F;
    auto d_M   = M;
    auto d_feq = feq;
    auto d_w   = w;
    auto d_w_bar= w_bar;
    auto d_e   = e;
    auto d_g   = g;
    int  d_Nx  = Nx;
    int  d_Ny  = Ny;
    int  d_Nz  = Nz;
    int  d_Np  = Np;
    int  d_Q   = Q;
    double d_pi= pi;
    double d_Lamda=Lamda;
    double d_tau_f= tau_f;
    double d_source= source;


    Kokkos::parallel_for("initialize",//calculating source term and equlibirum distribution fucntion for the preparation of evolution
    Kokkos::TeamPolicy<ExecSpace>((Nx+1)*(Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny + 1);
        const int j   = gid % (d_Ny + 1);

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
            d_g(i,j,z)=d_source*(9.0*d_pi*d_pi*d_pi*d_pi*Kokkos::cos(d_pi*double(i)/d_Np)*Kokkos::sin(d_pi*double(j)/d_Np)*Kokkos::sin(d_pi*double(z)/d_Np));
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        double weight = d_w(k);
                        if(k==0)
                    {
                        d_feq(i, j, z, k) = (1.0-weight) * d_U(i, j, z);
                        d_f(i, j, z, k)   = d_feq(i, j, z, k);
                    }
                        else 
                    {
                        d_feq(i,j,z,k)=weight*d_U(i,j,z);
                        d_f(i,j,z,k)=d_feq(i,j,z,k);
                    }
                    });
            });
    });
    Kokkos::parallel_for("collision",//collision
    Kokkos::TeamPolicy<ExecSpace>((Nx+1)*(Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny + 1);
        const int j   = gid % (d_Ny + 1);

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        double weight = d_w_bar(k);
                        d_M(i,j,z,k)=d_f(i,j,z,k)+(d_feq(i,j,z,k)-d_f(i,j,z,k))/d_tau_f+weight*(0.5-d_tau_f)*d_g(i,j,z)*d_Lamda/(d_Np*d_Np);
                    });
            });
    });
	Kokkos::parallel_for("streaming",//streaming
    Kokkos::TeamPolicy<ExecSpace>((Nx-1)*(Ny-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny - 1) + 1;
        const int j   = gid % (d_Ny - 1) + 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, d_Nz),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        int ip,jp,zp;
                        ip=i-d_e(k,0);
                        jp=j-d_e(k,1);
                        zp=z-d_e(k,2);
                        d_F(i,j,z,k)=d_M(ip,jp,zp,k);
                    });
            });
    });
	Kokkos::parallel_for("calculation of Marcroscopic variables",//calculation of Marcroscopic variables(solutions of the target PDEs)
    Kokkos::TeamPolicy<ExecSpace>((Nx-1)*(Ny-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny - 1) + 1;
        const int j   = gid % (d_Ny - 1) + 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, d_Nz),
            [=](int z)
            {
                d_U0(i,j,z)=d_U(i,j,z);
				d_U(i,j,z)=0;
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
					d_f(i,j,z,k)=d_F(i,j,z,k);
					d_U(i,j,z)+=d_f(i,j,z,k);
                    });
            });
    });
	Kokkos::parallel_for("calculation of feq ready for the boundary conditions",//calculation of feq ready for the boundary conditions
    Kokkos::TeamPolicy<ExecSpace>((Nx-1)*(Ny-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny - 1) + 1;
        const int j   = gid % (d_Ny - 1) + 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, d_Nz),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
					{
						d_feq(i,j,z,k)=(1.0-weight)*d_U(i,j,z);
						d_f(i,j,z,k)=d_feq(i,j,z,k);
					}
					else 
					{
						d_feq(i,j,z,k)=weight*d_U(i,j,z);
						d_f(i,j,z,k)=d_feq(i,j,z,k);
					}
                    });
            });
    });


////////////////////////////boundary conditions////////////////////////
	Kokkos::parallel_for("left and right walls",
    Kokkos::TeamPolicy<ExecSpace>((Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int j   = gid ;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
            //  d_U(0,j,z)=-3*pi*pi*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
			//  d_U(d_Nx,j,z)=3*pi*pi*sin(pi*double(j)/Np)*sin(pi*double(z)/Np);//Dirichlet boundary conditions
                d_U(0,j,z)=d_U(1,j,z);        //Neumann boundary conditions
				d_U(d_Nx,j,z)=d_U(d_Nx-1,j,z);//Neumann boundary conditions
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
				{
					d_feq(0,j,z,k)=(1.0-weight)*d_U(0,j,z);
					d_feq(d_Nx,j,z,k)=(1.0-weight)*d_U(d_Nx,j,z);
					d_f(0,j,z,k)=d_feq(0,j,z,k)+d_f(1,j,z,k)-d_feq(1,j,z,k);
					d_f(d_Nx,j,z,k)=d_feq(d_Nx,j,z,k)+d_f(d_Nx-1,j,z,k)-d_feq(d_Nx-1,j,z,k);
				}
				else 
				{
					d_feq(0,j,z,k)=weight*d_U(0,j,z);
					d_feq(d_Nx,j,z,k)=weight*d_U(d_Nx,j,z);
					d_f(0,j,z,k)=d_feq(0,j,z,k)+d_f(1,j,z,k)-d_feq(1,j,z,k);
					d_f(d_Nx,j,z,k)=d_feq(d_Nx,j,z,k)+d_f(d_Nx-1,j,z,k)-d_feq(d_Nx-1,j,z,k);
				}
                    });
            });
    });
	Kokkos::parallel_for("front and back",
    Kokkos::TeamPolicy<ExecSpace>((Nx-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank() + 1;
        const int i   = gid ;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {   			
            //  d_U(i,d_Ny,z)=0;//Dirichlet boundary conditions
			//  d_U(i,0,z)=0;//Dirichlet boundary conditions
                d_U(i,d_Ny,z)=0;//Neumann boundary conditions
				d_U(i,0,z)=0;   //Neumann boundary conditions
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
				{
					d_feq(i,0,z,k)=(1.0-weight)*d_U(i,0,z);
					d_feq(i,d_Ny,z,k)=(1.0-weight)*d_U(i,d_Ny,z);
					d_f(i,0,z,k)=d_feq(i,0,z,k)+d_f(i,1,z,k)-d_feq(i,1,z,k);
					d_f(i,d_Ny,z,k)=d_feq(i,d_Ny,z,k)+d_f(i,d_Ny-1,z,k)-d_feq(i,d_Ny-1,z,k);
				}
				else
				{
					d_feq(i,0,z,k)=weight*d_U(i,0,z);
					d_feq(i,d_Ny,z,k)=weight*d_U(i,d_Ny,z);
					d_f(i,0,z,k)=d_feq(i,0,z,k)+d_f(i,1,z,k)-d_feq(i,1,z,k);
					d_f(i,d_Ny,z,k)=d_feq(i,d_Ny,z,k)+d_f(i,d_Ny-1,z,k)-d_feq(i,d_Ny-1,z,k);
				}
                    });
            });
    });
	Kokkos::parallel_for("up and down",
    Kokkos::TeamPolicy<ExecSpace>((Nx-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank() + 1;
        const int i   = gid ;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1 , d_Ny),
            [=](int j)
            {
            //  d_U(i,j,0)=0;//Dirichlet boundary conditions
			//  d_U(i,j,d_Nz)=0;//Dirichlet boundary conditions
                d_U(i,j,0)=0;   //Neumann boundary conditions
				d_U(i,j,d_Nz)=0;//Neumann boundary conditions
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
				{
					d_feq(i,j,0,k)=(1.0-weight)*d_U(i,j,0);
					d_feq(i,j,d_Nz,k)=(1.0-weight)*d_U(i,j,d_Nz);
					d_f(i,j,0,k)=d_feq(i,j,0,k)+d_f(i,j,1,k)-d_feq(i,j,1,k);
					d_f(i,j,d_Nz,k)=d_feq(i,j,d_Nz,k)+d_f(i,j,d_Nz-1,k)-d_feq(i,j,d_Nz-1,k);
				}
				else 
				{
					d_feq(i,j,0,k)=weight*d_U(i,j,0);
					d_feq(i,j,d_Nz,k)=weight*d_U(i,j,d_Nz);
					d_f(i,j,0,k)=d_feq(i,j,0,k)+d_f(i,j,1,k)-d_feq(i,j,1,k);
					d_f(i,j,d_Nz,k)=d_feq(i,j,d_Nz,k)+d_f(i,j,d_Nz-1,k)-d_feq(i,j,d_Nz-1,k);
				}
                    });
            });
    });

////////////////////////////done////////////////////////////////
}
void LBM::evolution1()
{
    auto d_U   = U;
    auto d_U1   = U1;
    auto d_U0  = U0;
    auto d_f   = f;
    auto d_F   = F;
    auto d_M   = M;
    auto d_feq = feq;
    auto d_w   = w;
    auto d_w_bar= w_bar;
    auto d_e   = e;
    auto d_g   = g;
    int  d_Nx  = Nx;
    int  d_Ny  = Ny;
    int  d_Nz  = Nz;
    int  d_Np  = Np;
    int  d_Q   = Q;
    double d_pi= pi;
    double d_Lamda=Lamda;
    double d_tau_f= tau_f;



    Kokkos::parallel_for("initialize",
    Kokkos::TeamPolicy<ExecSpace>((Nx+1)*(Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny + 1);
        const int j   = gid % (d_Ny + 1);

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        double weight = d_w(k);
                        if(k==0)
                    {
                        d_feq(i, j, z, k) = (1.0-weight) * d_U1(i, j, z);
                        d_f(i, j, z, k)   = d_feq(i, j, z, k);
                    }
                        else 
                    {
                        d_feq(i,j,z,k)=weight*d_U1(i,j,z);
                        d_f(i,j,z,k)=d_feq(i,j,z,k);
                    }
                    });
            });
    });
    Kokkos::parallel_for("collision",//collision
    Kokkos::TeamPolicy<ExecSpace>((Nx+1)*(Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny + 1);
        const int j   = gid % (d_Ny + 1);

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        double weight = d_w_bar(k);
                        d_M(i,j,z,k)=d_f(i,j,z,k)+(d_feq(i,j,z,k)-d_f(i,j,z,k))/d_tau_f+weight*(0.5-d_tau_f)*d_U(i,j,z)*d_Lamda/(d_Np*d_Np);
                    });
            });
    });
	Kokkos::parallel_for("streaming",//streaming
    Kokkos::TeamPolicy<ExecSpace>((Nx-1)*(Ny-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny - 1) + 1;
        const int j   = gid % (d_Ny - 1) + 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, d_Nz),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                        if (k < 0 || k >= d_Q) return;
                        int ip,jp,zp;
                        ip=i-d_e(k,0);
                        jp=j-d_e(k,1);
                        zp=z-d_e(k,2);
                        d_F(i,j,z,k)=d_M(ip,jp,zp,k);
                    });
            });
    });
	Kokkos::parallel_for("calculation of Marcroscopic variables",//calculation of Marcroscopic variables
    Kokkos::TeamPolicy<ExecSpace>((Nx-1)*(Ny-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny - 1) + 1;
        const int j   = gid % (d_Ny - 1) + 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, d_Nz),
            [=](int z)
            {
                d_U0(i,j,z)=d_U1(i,j,z);
				d_U1(i,j,z)=0;
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
					d_f(i,j,z,k)=d_F(i,j,z,k);
					d_U1(i,j,z)+=d_f(i,j,z,k);
                    });
            });
    });
	Kokkos::parallel_for("calculation of feq ready for the boundary conditions",//calculation of feq ready for the boundary conditions
    Kokkos::TeamPolicy<ExecSpace>((Nx-1)*(Ny-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int i   = gid / (d_Ny - 1) + 1;
        const int j   = gid % (d_Ny - 1) + 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1, d_Nz),
            [=](int z)
            {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
					{
						d_feq(i,j,z,k)=(1.0-weight)*d_U1(i,j,z);
						d_f(i,j,z,k)=d_feq(i,j,z,k);
					}
					else 
					{
						d_feq(i,j,z,k)=weight*d_U1(i,j,z);
						d_f(i,j,z,k)=d_feq(i,j,z,k);
					}
                    });
            });
    });


////////////////////////////boundary conditions////////////////////////
	Kokkos::parallel_for("left and right walls",
    Kokkos::TeamPolicy<ExecSpace>((Ny+1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank();
        const int j   = gid ;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
                d_U1(0,j,z)=Kokkos::sin(d_pi*double(j)/d_Np)*Kokkos::sin(d_pi*double(z)/d_Np);    //Dirichlet boundary conditions
				d_U1(d_Nx,j,z)=-Kokkos::sin(d_pi*double(j)/d_Np)*Kokkos::sin(d_pi*double(z)/d_Np);//Dirichlet boundary conditions
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
				{
					d_feq(0,j,z,k)=(1.0-weight)*d_U1(0,j,z);
					d_feq(d_Nx,j,z,k)=(1.0-weight)*d_U1(d_Nx,j,z);
					d_f(0,j,z,k)=d_feq(0,j,z,k)+d_f(1,j,z,k)-d_feq(1,j,z,k);
					d_f(d_Nx,j,z,k)=d_feq(d_Nx,j,z,k)+d_f(d_Nx-1,j,z,k)-d_feq(d_Nx-1,j,z,k);
				}
				else 
				{
					d_feq(0,j,z,k)=weight*d_U1(0,j,z);
					d_feq(d_Nx,j,z,k)=weight*d_U1(d_Nx,j,z);
					d_f(0,j,z,k)=d_feq(0,j,z,k)+d_f(1,j,z,k)-d_feq(1,j,z,k);
					d_f(d_Nx,j,z,k)=d_feq(d_Nx,j,z,k)+d_f(d_Nx-1,j,z,k)-d_feq(d_Nx-1,j,z,k);
				}
                    });
            });
    });
	Kokkos::parallel_for("front and back",
    Kokkos::TeamPolicy<ExecSpace>((Nx-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank() + 1;
        const int i   = gid ;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, d_Nz + 1),
            [=](int z)
            {
                d_U1(i,d_Ny,z)=0;//Dirichlet boundary conditions
				d_U1(i,0,z)=0;   //Dirichlet boundary conditions
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
				{
					d_feq(i,0,z,k)=(1.0-weight)*d_U1(i,0,z);
					d_feq(i,d_Ny,z,k)=(1.0-weight)*d_U1(i,d_Ny,z);
					d_f(i,0,z,k)=d_feq(i,0,z,k)+d_f(i,1,z,k)-d_feq(i,1,z,k);
					d_f(i,d_Ny,z,k)=d_feq(i,d_Ny,z,k)+d_f(i,d_Ny-1,z,k)-d_feq(i,d_Ny-1,z,k);
				}
				else
				{
					d_feq(i,0,z,k)=weight*d_U1(i,0,z);
					d_feq(i,d_Ny,z,k)=weight*d_U1(i,d_Ny,z);
					d_f(i,0,z,k)=d_feq(i,0,z,k)+d_f(i,1,z,k)-d_feq(i,1,z,k);
					d_f(i,d_Ny,z,k)=d_feq(i,d_Ny,z,k)+d_f(i,d_Ny-1,z,k)-d_feq(i,d_Ny-1,z,k);
				}
                    });
            });
    });
	Kokkos::parallel_for("up and down",
    Kokkos::TeamPolicy<ExecSpace>((Nx-1), Kokkos::AUTO, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team)
    {
        const int gid = team.league_rank() + 1;
        const int i   = gid ;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 1 , d_Ny),
            [=](int j)
            {
                d_U1(i,j,0)=0;   //Dirichlet boundary conditions
				d_U1(i,j,d_Nz)=0;//Dirichlet boundary conditions
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, d_Q),
                    [=](int k)
                    {
                    if (k < 0 || k >= d_Q) return;
                    double weight = d_w(k);
					if(k==0)
				{
					d_feq(i,j,0,k)=(1.0-weight)*d_U1(i,j,0);
					d_feq(i,j,d_Nz,k)=(1.0-weight)*d_U1(i,j,d_Nz);
					d_f(i,j,0,k)=d_feq(i,j,0,k)+d_f(i,j,1,k)-d_feq(i,j,1,k);
					d_f(i,j,d_Nz,k)=d_feq(i,j,d_Nz,k)+d_f(i,j,d_Nz-1,k)-d_feq(i,j,d_Nz-1,k);
				}
				else 
				{
					d_feq(i,j,0,k)=weight*d_U1(i,j,0);
					d_feq(i,j,d_Nz,k)=weight*d_U1(i,j,d_Nz);
					d_f(i,j,0,k)=d_feq(i,j,0,k)+d_f(i,j,1,k)-d_feq(i,j,1,k);
					d_f(i,j,d_Nz,k)=d_feq(i,j,d_Nz,k)+d_f(i,j,d_Nz-1,k)-d_feq(i,j,d_Nz-1,k);
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
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<h_U1(20,j,z)<<std::endl;
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
			out<<double(j)/Np<<" "<<double(z)/Np<<" "<<h_U1(80,j,z)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<h_U1(i,20,z)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(z)/Np<<" "<<h_U1(i,80,z)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<h_U1(i,j,20)<<std::endl;
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
			out<<double(i)/Np<<" "<<double(j)/Np<<" "<<h_U1(i,j,80)<<std::endl;
		}
	}
}
void LBM::Error()
{
    auto d_temp1 = temp1;
    auto d_temp2 = temp2;
    auto d_U   = U;
    auto d_U0  = U0;
    h_temp1() = 0.0;
    h_temp2() = 0.0;
    Kokkos::deep_copy(d_temp1, h_temp1);
    Kokkos::deep_copy(d_temp2, h_temp2);
    Kokkos::parallel_reduce("error",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {Nx,Ny,Nz}),
        KOKKOS_LAMBDA(int i, int j, int z, double& l2sum, double& l2ref) {
            double du = d_U(i,j,z) - d_U0(i,j,z);
            l2sum += du * du;
            l2ref += d_U(i,j,z) * d_U(i,j,z);
        }, Kokkos::Sum<Precision,Device>(d_temp1), Kokkos::Sum<Precision,Device>(d_temp2));
    Kokkos::fence();
    Kokkos::deep_copy(h_temp1, d_temp1);
    Kokkos::deep_copy(h_temp2, d_temp2);
    h_err() = static_cast<Precision>(std::sqrt(h_temp1()) / (std::sqrt(h_temp2()) + 1e-30));
}
void LBM::Error1()
{
    auto d_temp1 = temp1;
    auto d_temp2 = temp2;
    auto d_U1  = U1;
    auto d_U0  = U0;
    h_temp1() = 0.0;
    h_temp2() = 0.0;
    Kokkos::deep_copy(d_temp1, h_temp1);
    Kokkos::deep_copy(d_temp2, h_temp2);
    Kokkos::parallel_reduce("error",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {Nx,Ny,Nz}),
        KOKKOS_LAMBDA(int i, int j, int z, double& l2sum, double& l2ref) {
            double du = d_U1(i,j,z) - d_U0(i,j,z);
            l2sum += du * du;
            l2ref += d_U1(i,j,z) * d_U1(i,j,z);
        }, Kokkos::Sum<Precision,Device>(d_temp1), Kokkos::Sum<Precision,Device>(d_temp2));
    Kokkos::deep_copy(h_temp1, d_temp1);
    Kokkos::deep_copy(h_temp2, d_temp2);
    h_err() = static_cast<Precision>(std::sqrt(h_temp1()) / (std::sqrt(h_temp2()) + 1e-30));
}
}
