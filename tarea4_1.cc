#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <unistd.h>

using namespace std;

int main(int argc, char *argv[]) {
    int rank, size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    long long total_points = 100000000;
    long long local_points, points_in_circle = 0;
    long long global_points_in_circle;
    double pi;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &namelen);
    
    cout << "[Process " << rank << "/" << size << "] Running on: " << processor_name << " (PID: " << getpid() << ")" << endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Dividir trabajo entre procesos
    local_points = total_points / size;
    
    if(rank == 0) {
        cout << "Calculating π using Monte Carlo method" << endl;
        cout << "_______________________________________" << endl;
        cout << "Total points: " << total_points << endl;
        cout << "Points per process: " << local_points << endl;
        cout << "Number of processes: " << size << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    cout << "[Process " << rank << " on " << processor_name << "] Starting calculation..." << endl;
    
    // Generador de números aleatorios
    random_device rd;
    mt19937 gen(rd() + rank);
    uniform_real_distribution<double> dis(0.0, 1.0);
    
    // Generar puntos aleatorios y contar los que caen dentro del círculo
    for(long long i = 0; i < local_points; i++) {
        double x = dis(gen);
        double y = dis(gen);
        double distance = x * x + y * y;
        
        if(distance <= 1.0) {
            points_in_circle++;
        }
    }
    
    cout << "[Process " << rank << " on " << processor_name << "] Points in circle: " << points_in_circle << " / " << local_points << " (" << fixed << setprecision(2) << (100.0 * points_in_circle / local_points) << "%)" << endl;
    
    // Comunicación colectiva: sumar resultados de todos los procesos
    MPI_Reduce(&points_in_circle, &global_points_in_circle, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    // Proceso 0 calcula y muestra el resultado final
    if(rank == 0) {
        pi = 4.0 * global_points_in_circle / total_points;
        cout << "Total points in circle: " << global_points_in_circle << endl;
        cout << fixed << setprecision(10);
        cout << "Estimated π: " << pi << endl;
        cout << "Actual π:    " << M_PI << endl;
        cout << "Error:       " << (pi - M_PI) << " (" << setprecision(6) << (100.0 * abs(pi - M_PI) / M_PI) << "%)" << endl;
        cout << setprecision(4);
        cout << "Time:        " << (end_time - start_time) << " seconds" << endl;
        cout << "Speedup:     " << setprecision(2) << (double)size << endl;
    }
    
    MPI_Finalize();
    return 0;
}