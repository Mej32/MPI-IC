#ifndef __IMAGE__H__
#define __IMAGE__H__
#include <vector>
#include <memory>
#include <iostream>
#include "assert.h"
#include <string>
#include "mpi.h" 

template <typename T> class Block;

template <typename T> class Image{
public:
  int width, height, channels;
  std::shared_ptr<T[]> matrix;
  void release();
  Image();
  Image(int width, int height, int channels);
  Image(const Image<T> &a);
  ~Image();
  Image<T> operator=(const Image<T>& other);
  Image<T> operator*(const Image<T>& other) const;
  Image<T> operator*(float scalar) const;
  Image<T> operator+(const Image<T>& other) const;
  Image<T> operator+(float scalar) const;
  T get(int row, int col, int channel) const;
  void set(int row, int col, int channel, T value);
  template <typename S> Image<S> convert() const;
  Image<T> to_grayscale() const;
  Image<T> abs() const;
  Image<float> normalized() const;
  Image<T> convolution(const Image<float> &kernel) const;
  std::vector<Block<T>> get_blocks(int block_size=8);
  void sync_data();
  void get_row_range(int &start, int &end) const;
};

Image<unsigned char> load_from_file(const std::string &filename);
void save_to_file(const std::string &filename, const Image<unsigned char> &image, int quality=100);

template <typename T> class Block{
public:
    int i, j, size, depth, rowsize;
    Image<T> *matrix;
    T get_pixel(int row, int col, int channel) const;
    void set_pixel(int row, int col, int channel, T value);
};

template <class T> T Block<T>::get_pixel(int row, int col, int channel) const {
    assert(row>=0 && row<size && col>=0 && col<size);
    return matrix->get(row+j, col+i, channel);
}
template <class T> void Block<T>::set_pixel(int row, int col, int channel, T value) {
    assert(row>=0 && row<size && col>=0 && col<size);
    return matrix->set(row+j, col+i, channel, value);
}


template <class T> void Image<T>::get_row_range(int &start, int &end) const {
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    
    int rows_per_proc = height / procs;
    int remainder = height % procs;
    
    if (rank < remainder) {
        start = rank * (rows_per_proc + 1);
        end = start + rows_per_proc + 1;
    } else {
        start = rank * rows_per_proc + remainder;
        end = start + rows_per_proc;
    }
}


template <class T> void Image<T>::sync_data() {
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    
    if (procs == 1) return;
    
    MPI_Datatype mpi_type;
    if (std::is_same<T, unsigned char>::value) mpi_type = MPI_UNSIGNED_CHAR;
    else if (std::is_same<T, float>::value) mpi_type = MPI_FLOAT;
    else if (std::is_same<T, int>::value) mpi_type = MPI_INT;
    else {
        std::cerr << "Tipo no soportado para sync_data" << std::endl;
        return;
    }
    
    std::vector<int> recvcounts(procs);
    std::vector<int> displs(procs);
    
    int row_bytes = width * channels;
    
    for (int p = 0; p < procs; p++) {
        int p_start, p_end;
        int rows_per_proc = height / procs;
        int remainder = height % procs;
        
        if (p < remainder) {
            p_start = p * (rows_per_proc + 1);
            p_end = p_start + rows_per_proc + 1;
        } else {
            p_start = p * rows_per_proc + remainder;
            p_end = p_start + rows_per_proc;
        }
        
        recvcounts[p] = (p_end - p_start) * row_bytes;
        displs[p] = p_start * row_bytes;
    }
    
    MPI_Allgatherv(MPI_IN_PLACE, 0, mpi_type,matrix.get(), recvcounts.data(), displs.data(), mpi_type,MPI_COMM_WORLD);
}

template <class T> Image<T>::Image() {
    matrix = nullptr;
}

template <class T> Image<T>::Image(int width, int height, int channels) {
    this->width = width;
    this->height = height;
    this->channels = channels;
    matrix = std::shared_ptr<T[]>(new T[height*width*channels]());
}

template <class T> Image<T>::Image(const Image<T> &a) {
    width = a.width;
    height = a.height;
    channels = a.channels;
    matrix = a.matrix;
}

template <class T> Image<T>::~Image() {
    release();
}

template <class T> Image<T> Image<T>::operator=(const Image<T> &a) {
    if (this == &a) return *this;
    release();
    width = a.width;
    height = a.height;
    channels = a.channels;
    matrix = a.matrix;
    return *this;
}

template <class T> void Image<T>::release() {
    matrix = nullptr;
}

template <class T> T Image<T>::get(int row, int col, int channel) const{
    return matrix[row*width*channels + col*channels + channel];
}

template <class T> void Image<T>::set(int row, int col, int channel, T value) {
    matrix[row*width*channels + col*channels + channel] = value;
}

template <class T> Image<T> Image<T>::operator*(const Image<T>& other) const {
    assert(width == other.width && height == other.height && channels == other.channels);
    Image<T> new_image(width, height, channels);
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j, i, c, this->get(j, i, c) * other.get(j, i, c));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> Image<T> Image<T>::operator*(float scalar) const {
    Image<T> new_image(width, height, channels);
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j, i, c, (T)(this->get(j, i, c)*scalar));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> Image<T> Image<T>::operator+(const Image<T>& other) const {
    assert(width == other.width && height == other.height && channels == other.channels);
    Image<T> new_image(width, height, channels);
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j,i,c, this->get(j, i,c)+other.get(j, i, c));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> Image<T> Image<T>::operator+(float scalar) const {
    Image<T> new_image(width, height, channels);
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j, i, c, ((T)this->get(j, i, c)+scalar));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> Image<T> Image<T>::abs() const {
    Image<T> new_image(width, height, channels);
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j, i, c, (T)std::abs(this->get(j,i,c)));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> Image<T> Image<T>::convolution(const Image<float> &kernel) const {
    assert(kernel.width%2 != 0 && kernel.height%2 != 0 && kernel.width == kernel.height && kernel.channels==1);
    int kernel_size = kernel.width;
    Image<T> convolved(width, height, channels);
    
    int start_row, end_row;
    convolved.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                float sum = 0.0;
                for(int u=0; u<kernel_size; u++) {
                    for(int v=0; v<kernel_size; v++) {
                        int s = (j + u - kernel_size/2)%height;
                        int t = (i + v - kernel_size/2)%width;
                        if (s < 0 || s >= height || t < 0 || t >= width)
                            continue;
                        sum += (this->get(s, t, c) * kernel.get(u,v, 0));
                    }
                }
                convolved.set(j, i, c, (T)(sum/(kernel_size*kernel_size)));
            }
        }
    }
    
    convolved.sync_data();
    return convolved;
}

template <class T> template <typename S> Image<S> Image<T>::convert() const {
    Image<S> new_image(width, height, channels);
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j, i, c, (S)this->get(j, i, c));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> Image<T> Image<T>::to_grayscale() const {
    if (channels == 1) return convert<T>();
    Image<T> image(width, height, 1);
    
    int start_row, end_row;
    image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            image.set(j, i, 0, (T)((0.299 * this->get(j, i, 0) + (0.587 * this->get(j, i, 1)) + (0.114 * this->get(j,i,2)))));
        }
    }
    
    image.sync_data();
    return image;
}

template <class T> Image<float> Image<T>::normalized() const {
    Image<float> new_image(width, height, channels);
    float max_value = -999999999;
    float min_value = 999999999;
    
    int start_row, end_row;
    new_image.get_row_range(start_row, end_row);
    
    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                if (this->get(j,i,c) > max_value) max_value = this->get(j,i,c);
                if (this->get(j,i,c) < min_value) min_value = this->get(j,i,c);
            }
        }    
    }

    float global_max, global_min;
    MPI_Allreduce(&max_value, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&min_value, &global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    for(int j=start_row; j<end_row; j++) {
        for(int i=0; i<width; i++) {
            for(int c=0; c<channels; c++) {
                new_image.set(j,i,c, (this->get(j, i, c)-global_min) / (global_max - global_min));
            }
        }    
    }
    
    new_image.sync_data();
    return new_image;
}

template <class T> std::vector<Block<T>> Image<T>::get_blocks(int block_size) {
    int depth = channels;
    assert(width % block_size == 0 || height % block_size == 0);
    std::vector<Block<T>> blocks;
    
    for (int row=0; row<height; row+=block_size) {
        for(int col=0; col<width; col+=block_size) {
            Block<T> b;
            b.i=col;
            b.j=row;
            b.size=block_size;
            b.rowsize=width*channels;
            b.matrix=this;
            b.depth=depth;
            blocks.push_back(b);
        }
    }
    return blocks;
}

#endif