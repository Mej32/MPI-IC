#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include "mpi.h"

Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    switch(size){
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();
}

Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size, int rank) {
    auto begin = std::chrono::steady_clock::now();
    if(rank == 0) {
        std::cout<<"Computing SRM "<<kernel_size<<"x"<<kernel_size<<"..."<<std::endl;
    }
    
    Image<float> srm = image.to_grayscale().convert<float>();
    srm = srm.convolution(get_srm_kernel(kernel_size));
    srm = srm.abs().normalized();
    srm = srm * 255;
    Image<unsigned char> result = srm.convert<unsigned char>();
    
    auto end = std::chrono::steady_clock::now();
    if(rank == 0) {
        std::cout<<"SRM elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    }
    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert, int rank) {
    auto begin = std::chrono::steady_clock::now();
    if(rank == 0) {
        std::cout<<"Computing"; 
        if (invert) std::cout<<" inverse";
        else std::cout<<" direct";
        std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;
    }
    
    int rank_mpi, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    
    // Imagen de ENTRADA (original)
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> input_blocks = grayscale.get_blocks(block_size);
    
    // Imagen de SALIDA (en ceros para acumular resultados)
    Image<float> output(grayscale.width, grayscale.height, 1);
    for(int i=0; i<output.width * output.height; i++) {
        output.matrix.get()[i] = 0.0f;
    }
    std::vector<Block<float>> output_blocks = output.get_blocks(block_size);
    
    // Cada proceso trabaja en bloques diferentes
    for(int i=rank_mpi; i<input_blocks.size(); i+=procs){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        
        // DCT directo sobre el bloque de ENTRADA
        dct::direct(dctBlock, input_blocks[i], 0);
        
        if (invert) {
            // Filtrar frecuencias altas
            for(int k=0; k<block_size/2; k++)
                for(int l=0; l<block_size/2; l++)
                    dctBlock[k][l] = 0.0;
            // DCT inverso al bloque de SALIDA
            dct::inverse(output_blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            // Asignar resultado al bloque de SALIDA
            dct::assign(dctBlock, output_blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }
    
    // Sincronización: sumar los bloques procesados por cada proceso
    if (procs > 1) {
        MPI_Allreduce(MPI_IN_PLACE, output.matrix.get(), 
                      output.width * output.height, 
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    
    Image<unsigned char> result = output.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    if(rank == 0) {
        std::cout<<"DCT elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    }
    return result;
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality, int rank){
    if(rank == 0) {
        std::cout<<"Computing ELA..."<<std::endl;
    }
    auto begin = std::chrono::steady_clock::now();
    
    Image<unsigned char> grayscale = image.to_grayscale();
    
    // Solo el proceso 0 guarda y carga
    if(rank == 0) {
        save_to_file("imagenes/_temp.jpg", grayscale, quality);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    Image<float> compressed = load_from_file("imagenes/_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();
    
    auto end = std::chrono::steady_clock::now();
    if(rank == 0) {
        std::cout<<"ELA elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    }
    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc == 1) {
        if (rank == 0) std::cerr << "Error de comando, falta la imagen" << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
    system("mkdir -p imagenes");
    }
    MPI_Barrier(MPI_COMM_WORLD);  // Esperar a que se cree la carpeta

    auto total_begin = std::chrono::steady_clock::now();
    Image<unsigned char> image;
    int img_info[3];
    
    // Proceso 0 carga la imagen
    if (rank == 0) {
        image = load_from_file(argv[1]); 
        img_info[0] = image.width;
        img_info[1] = image.height;
        img_info[2] = image.channels;
    }
    
    // Difundir información de la imagen a todos los procesos
    MPI_Bcast(img_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Informacion a repartir
    if (rank != 0) {
        image = Image<unsigned char>(img_info[0], img_info[1], img_info[2]);
    }
    
    // Difundir los datos de la imagen a todos
    MPI_Bcast(image.matrix.get(), img_info[0] * img_info[1] * img_info[2], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    int block_size = 8;
    Image<unsigned char> srm3 = compute_srm(image, 3, rank);
    if (rank == 0) {
        save_to_file("imagenes/srm_kernel_3x3.png", srm3);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Image<unsigned char> srm5 = compute_srm(image, 5, rank);
    if (rank == 0) {
        save_to_file("imagenes/srm_kernel_5x5.png", srm5);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  
    Image<unsigned char> ela = compute_ela(image, 90, rank);
    if (rank == 0) {
        save_to_file("imagenes/ela.png", ela);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Image<unsigned char> dct_inv = compute_dct(image, block_size, true, rank);
    if (rank == 0) {
        save_to_file("imagenes/dct_invert.png", dct_inv);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Image<unsigned char> dct_dir = compute_dct(image, block_size, false, rank);
    if (rank == 0) {
        save_to_file("imagenes/dct_direct.png", dct_dir);
    }
    
    auto total_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        std::cout << "TOTAL TIME MPI: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_begin).count() << "ms" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}