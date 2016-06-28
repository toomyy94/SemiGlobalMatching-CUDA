
// Based on CUDA SDK template from NVIDIA
// sgm algorithm adapted from http://lunokhod.org/?p=1403

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include <limits>
#include <algorithm>

// includes, project
#include <cutil_inline.h>

#define MMAX_BRIGHTNESS 255

#define PENALTY1 15
#define PENALTY2 100

#define COSTS(i,j,d)              costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define ACCUMULATED_COSTS(i,j,d)  accumulated_costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define LEFT_IMAGE(i,j)           left_image[(i)+(j)*nx]
#define RIGHT_IMAGE(i,j)          right_image[(i)+(j)*nx]
#define DISP_IMAGE(i,j)           disp_image[(i)+(j)*nx]

#define MMAX(a,b) (((a)>(b))?(a):(b))
#define MMIN(a,b) (((a)<(b))?(a):(b))
#define MYASSERT(condition) if (!(condition)) return;

/* function headers */


void determine_costs(const int *left_image, const int *right_image, int *costs, const int nx, const int ny, const int disp_range);

void evaluate_path( const int *prior, const int* local,
                    int path_intensity_gradient, int *curr_cost, 
                    const int nx, const int ny, const int disp_range );

__device__ void devEvaluate_path(const int *prior, const int *local, int path_intensity_gradient, int *curr_cost , const int nx, const int ny, const int disp_range);


void iterate_direction_dirxpos(const int dirx, const int *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction_dirypos(const int diry, const int *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction_dirxneg(const int dirx, const int *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction_diryneg(const int diry, const int *left_image,
                               const int* costs, int *accumulated_costs, 
                               const int nx, const int ny, const int disp_range );

void iterate_direction( const int dirx, const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) ;

void inplace_sum_views( int * im1, const int * im2, 
                        const int nx, const int ny, const int disp_range ) ;

int find_min_index( const int *v, const int dist_range ) ;

void create_disparity_view( const int *accumulated_costs , int * disp_image, int nx, int ny) ;

void sgmHost(   const int *h_leftIm, const int *h_rightIm, 
                int *h_dispIm, 
                const int w, const int h, const int disp_range );

void sgmDevice( const int *h_leftIm, const int *h_rightIm, 
                int *h_dispImD, 
                const int w, const int h, const int disp_range );

void usage(char *command);


/* functions code */

void determine_costs(const int *left_image, const int *right_image, int *costs, const int nx, const int ny, const int disp_range){

	std::fill(costs, costs+nx*ny*disp_range, 255u); 

	for ( int j = 0; j < ny; j++ ) {
		for ( int d = 0; d < disp_range; d++ ) {
			for ( int i = d; i < nx; i++ ) {
				COSTS(i,j,d) = abs( LEFT_IMAGE(i,j) - RIGHT_IMAGE(i-d,j) );
			}
		}
	}
}



__global__ void devDetermine_costs(const int *left_image, const int *right_image, int *costs, const int nx, const int ny, const int disp_range) {
  
  int j = threadIdx.y + blockIdx.y * blockDim.y; // verificar se j < ny
  MYASSERT(j < ny);
  int i = threadIdx.x + blockIdx.x * blockDim.x; //verificar se i < nx
  MYASSERT(i < nx);
  int d = threadIdx.z + blockIdx.z * blockDim.z; //verificar se d < disp_range
  MYASSERT(d < disp_range);

  COSTS(i,j,d)=255;
    if (i >= d){
      COSTS(i,j,d) = abs( LEFT_IMAGE(i,j) - RIGHT_IMAGE(i-d,j) );
    }

}

// ADD two cost images 
void inplace_sum_views( int * im1, const int * im2, 
                        const int nx, const int ny, const int disp_range ) 
{
    int *im1_init = im1;
    while ( im1 != (im1_init + (nx*ny*disp_range)) ) {
      *im1 += *im2;
      im1++;
      im2++;
  }
}

__global__ void devInplace_sum_views( int * im1, const int * im2, const int nx, const int ny, const int disp_range ) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  MYASSERT(x < nx);
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  MYASSERT(y < ny);
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  MYASSERT(z < disp_range);
  
  int id = x + y * nx + z * nx * ny;
  im1[id] += im2[id];
}


//////////////////////////////////////////////////////////////////////////////////
void iterate_direction_dirxpos(const int dirx, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) {
const int WIDTH = nx;
const int HEIGHT = ny;

for ( int j = 0; j < HEIGHT; j++ ) {
  for ( int i = 0; i < WIDTH; i++ ) {
    if(i==0) {
      for ( int d = 0; d < disp_range; d++ ) {
        ACCUMULATED_COSTS(0,j,d) += COSTS(0,j,d);
      }
    }
    else {
      evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
              &COSTS(i,j,0),
              abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)) ,
              &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range);
    }
  }
}
}


__global__ void devIterate_direction_dirxpos(const int dirx, const int *left_image, const int* costs, int *accumulated_costs,  const int nx, const int ny, const int disp_range ) {
  
  const int WIDTH = nx;
  //const int HEIGHT = ny;

  int j = blockIdx.y;
   int d = threadIdx.x;
  
  if(j < ny && d < disp_range){
    for ( int i = 0; i < WIDTH; i++ ) {
      //if(i<WIDTH /*&& j< HEIGHT*/) {
        if(i==0) {
          //for ( int d = 0; d < disp_range; d++ ) {
            ACCUMULATED_COSTS(0,j,d) += COSTS(0,j,d);
          //}
        }
        else {
          devEvaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0), &COSTS(i,j,0), abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)), &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range);
        }
        __syncthreads();
      //}
  }
  }

}
//////////////////////////////////////////////////////////////////////////////////////////7
void iterate_direction_dirypos(const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int i = 0; i < WIDTH; i++ ) {
          for ( int j = 0; j < HEIGHT; j++ ) {
              if(j==0) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,0,d) += COSTS(i,0,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
              }
          }
      }
}


__global__ void devIterate_direction_dirypos(const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    //const int WIDTH = nx;
    const int HEIGHT = ny;

    int i = blockIdx.y;
     int d = threadIdx.x;

    if(i < nx && d < disp_range)
    for ( int j = 0; j < HEIGHT; j++ ) {
        if(j==0) {
            //for ( int d = 0; d < disp_range; d++ ) {
                ACCUMULATED_COSTS(i,0,d) += COSTS(i,0,d);
            //}
        }
        else {
            devEvaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                           &COSTS(i,j,0),
                           abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                           &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
        }
        __syncthreads();

    }



}
////////////////////////////////////////////////////////////////////////////////////////7

void iterate_direction_dirxneg(const int dirx, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

      for ( int j = 0; j < HEIGHT; j++ ) {
          for ( int i = WIDTH-1; i >= 0; i-- ) {
              if(i==WIDTH-1) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(WIDTH-1,j,d) += COSTS(WIDTH-1,j,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
                                 &COSTS(i,j,0),
                                 abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)),
                                 &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
              }
          }
      }
}

__global__  void devIterate_direction_dirxneg(const int dirx, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{

    const int WIDTH = nx;
    int j = blockIdx.y;
     int d = threadIdx.x;

    if (j<ny && d < disp_range){
            //if(i >=0 && i<=WIDTH-1 /*&& j< HEIGHT*/) {
          for ( int i = WIDTH-1; i >= 0; i-- ) {
                  if(i==WIDTH-1) {
                      //for ( int d = 0; d < disp_range; d++ ) {
                          ACCUMULATED_COSTS(WIDTH-1,j,d) += COSTS(WIDTH-1,j,d);
                      //}
                  }
                  else {
                      devEvaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
                                     &COSTS(i,j,0),
                                     abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)),
                                     &ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
                  }
                  __syncthreads();
          }
    }





}

////////////////////////////////////////////////////////////////////////////////////////

void iterate_direction_diryneg(const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    const int WIDTH = nx;
    const int HEIGHT = ny;

   

      for ( int i = 0; i < WIDTH; i++ ) {
          for ( int j = HEIGHT-1; j >= 0; j-- ) {
              if(j==HEIGHT-1) {
                  for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,HEIGHT-1,d) += COSTS(i,HEIGHT-1,d);
                  }
              }
              else {
                  evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                           &COSTS(i,j,0),
                           abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                           &ACCUMULATED_COSTS(i,j,0) , nx, ny, disp_range);
             }
         }
      }


}


__global__  void devIterate_direction_diryneg(const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
     const int HEIGHT = ny;
 
     int i = blockIdx.y;
     int d = threadIdx.x;
     //int j = blockIdx.y*blockDim.y + threadIdx.y;
      //for ( int i = 0; i < WIDTH; i++ ) {
      if(i < nx && d < disp_range){
          for ( int j = HEIGHT-1; j >= 0; j-- ) {
 
              if(j==HEIGHT-1) {
                  //for ( int d = 0; d < disp_range; d++ ) {
                      ACCUMULATED_COSTS(i,HEIGHT-1,d) += COSTS(i,HEIGHT-1,d);
                  //}
              }
              else {
                  devEvaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
                           &COSTS(i,j,0),
                           abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
                           &ACCUMULATED_COSTS(i,j,0) , nx, ny, disp_range);
             }
             __syncthreads();
         }
       }
     
}
 
void iterate_direction( const int dirx, const int diry, const int *left_image,
                        const int* costs, int *accumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    // Walk along the edges in a clockwise fashion
    if ( dirx > 0 ) {
      // LEFT MOST EDGE
      // Process every pixel along this edge
      iterate_direction_dirxpos(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
    } 
    else if ( diry > 0 ) {
      // TOP MOST EDGE
      // Process every pixel along this edge only if dirx ==
      // 0. Otherwise skip the top left most pixel
      iterate_direction_dirypos(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
    } 
    else if ( dirx < 0 ) {
      // RIGHT MOST EDGE
      // Process every pixel along this edge only if diry ==
      // 0. Otherwise skip the top right most pixel
      iterate_direction_dirxneg(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
    } 
    else if ( diry < 0 ) {
      // BOTTOM MOST EDGE
      // Process every pixel along this edge only if dirx ==
      // 0. Otherwise skip the bottom left and bottom right pixel
      iterate_direction_diryneg(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
    }
}

void devIterate_direction( const int dirx, const int diry, const int *devLeft_image,
                        const int* devCosts, int *devAccumulated_costs, 
                        const int nx, const int ny, const int disp_range ) 
{
    // Walk along the edges in a clockwise fashion



  int block_x = disp_range;
  int block_y = 1; 
  dim3 dimBlock(block_x,block_y);
  dim3 dimGridy(1,nx);
  dim3 dimGridx(1,ny);
  
  //int *devLeft_image, *devCosts, *devAccumulated_costs; 

  //cudaMalloc((void **)&devLeft_image, nx*ny*sizeof(int));
  //cudaMalloc((void **)&devCosts, nx*ny*sizeof(int)*disp_range);
  //cudaMalloc((void **)&devAccumulated_costs, nx*ny*sizeof(int)*disp_range);

  //cudaMemcpy(devLeft_image, left_image, nx*ny*sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(devCosts, costs, nx*ny*sizeof(int)*disp_range, cudaMemcpyHostToDevice);
  //cudaMemcpy(devAccumulated_costs, accumulated_costs, nx*ny*sizeof(int)*disp_range, cudaMemcpyHostToDevice);

    if ( dirx > 0 ) {
      devIterate_direction_dirxpos<<<dimGridx, dimBlock>>> (dirx,devLeft_image,devCosts,devAccumulated_costs, nx, ny, disp_range);
    }
    else if ( diry > 0 ) {
      devIterate_direction_dirypos<<<dimGridy, dimBlock>>> (diry,devLeft_image,devCosts,devAccumulated_costs, nx, ny, disp_range);
    } 
    else if ( dirx < 0 ) {
      devIterate_direction_dirxneg<<<dimGridx, dimBlock>>> (dirx,devLeft_image,devCosts,devAccumulated_costs, nx, ny, disp_range);
    } 
    else if ( diry < 0 ) {
      devIterate_direction_diryneg<<<dimGridy, dimBlock>>> (diry,devLeft_image,devCosts,devAccumulated_costs, nx, ny, disp_range);
    }

  //cudaMemcpy(accumulated_costs, devAccumulated_costs, nx*ny*sizeof(int)*disp_range, cudaMemcpyDeviceToHost);

  /* libertar a memoria do device que ja nao vai ser usada */
  //cudaFree(devCosts);
  //cudaFree(devLeft_image);
  //cudaFree(devAccumulated_costs);
  
}

int find_min_index( const int *v, const int disp_range ) 
{
    int min = std::numeric_limits<int>::max();
    int minind = -1;
    for (int d=0; d < disp_range; d++) {
         if(v[d]<min) {
              min = v[d];
              minind = d;
         }
    }
    return minind;
}

void evaluate_path(const int *prior, const int *local,
                   int path_intensity_gradient, int *curr_cost , 
                   const int nx, const int ny, const int disp_range) 
{
  memcpy(curr_cost, local, sizeof(int)*disp_range);

  for ( int d = 0; d < disp_range; d++ ) {
    int e_smooth = std::numeric_limits<int>::max();
    for ( int d_p = 0; d_p < disp_range; d_p++ ) {
      if ( d_p - d == 0 ) {
        // No penality
        e_smooth = MMIN(e_smooth,prior[d_p]);
      } else if ( abs(d_p - d) == 1 ) {
        // Small penality
        e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
      } else {
        // Large penality
        e_smooth =
          MMIN(e_smooth,prior[d_p] +
                   MMAX(PENALTY1,
                            path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
      }
    }
    curr_cost[d] += e_smooth;
  }

  int min = std::numeric_limits<int>::max();
  for ( int d = 0; d < disp_range; d++ ) {
        if (prior[d]<min) min=prior[d];
  }
  for ( int d = 0; d < disp_range; d++ ) {
        curr_cost[d]-=min;
  }
}

__device__ void devEvaluate_path(const int *prior, const int *local, int path_intensity_gradient, int *curr_cost , const int nx, const int ny, const int disp_range) 
{

  int d = threadIdx.x;
  curr_cost[d] = local[d];
  //__syncthreads();
  //for ( int d = 0; d < disp_range; d++ ) {
    int e_smooth = 2147483647;
    for ( int d_p = 0; d_p < disp_range; d_p++ ) {
      if ( d_p - d == 0 ) {
        // No penality
        e_smooth = MMIN(e_smooth,prior[d_p]);
      } else if ( abs(d_p - d) == 1 ) {
        // Small penality
        e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
      } else {
        // Large penality
        e_smooth =
          MMIN(e_smooth,prior[d_p] +
                   MMAX(PENALTY1,
                            path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
      }
    }
    
    curr_cost[d] += e_smooth;
    __syncthreads();
  //}

  int min = 2147483647;
  for ( int k = 0; k < disp_range; k++ ) {
        if (prior[k]<min) min=prior[k];
  }
  //for ( int d = 0; d < disp_range; d++ ) {
      //__syncthreads();
        curr_cost[d]-=min;
  //}

}

void create_disparity_view( const int *accumulated_costs , int * disp_image, 
                            const int nx, const int ny, const int disp_range) 
{
  for ( int j = 0; j < ny; j++ ) {
    for ( int i = 0; i < nx; i++ ) {
      DISP_IMAGE(i,j) =
        4 * find_min_index( &ACCUMULATED_COSTS(i,j,0), disp_range );
    }
  }
}


 

/*
 * Links:
 * http://www.dlr.de/rmc/rm/en/desktopdefault.aspx/tabid-9389/16104_read-39811/
 * http://lunokhod.org/?p=1356
 */

//ISTO É O HOST
// sgm code to run on the host
void sgmHost(   const int *h_leftIm, const int *h_rightIm,  int *h_dispIm,  const int w, const int h, const int disp_range) {
	const int nx = w;
	const int ny = h;

	// Processing all costs. W*H*D. D= disp_range
	int *costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	
	if (costs == NULL) { 
		fprintf(stderr, "sgm_cuda:"
		" Failed memory allocation(s).\n");
		exit(1);
	}

	determine_costs(h_leftIm, h_rightIm, costs, nx, ny, disp_range);

	int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	
	if (accumulated_costs == NULL || dir_accumulated_costs == NULL) { 
		fprintf(stderr, "sgm_cuda:"
		" Failed memory allocation(s).\n");
		exit(1);
	}

	int dirx=0,diry=0;
	for(dirx=-1; dirx<2; dirx++) {
		if(dirx==0 && diry==0) continue;
		std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
		iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
		inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
	}
	dirx=0;
	for(diry=-1; diry<2; diry++) {
		if(dirx==0 && diry==0) continue;
		std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
		iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
		inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
	}

	free(costs);
	free(dir_accumulated_costs);

	create_disparity_view( accumulated_costs, h_dispIm, nx, ny, disp_range );

	free(accumulated_costs);
}
//ISTO É O DEVICE
// sgm code to run on the GPU
void sgmDevice( const int *h_leftIm, const int *h_rightIm,  int *h_dispIm, const int w, const int h, const int disp_range ){

  const int nx = w;
  const int ny = h;

  /*adicionado */
  const int size = nx * ny;
  const int memsize = size * sizeof(int);


  /*cuda arrays costs */ 
  int *devCosts; //, *devAccumulated_costs, *devDir_accumulated_costs; //tarefa 1 e 3
  
  cudaMalloc((void **)&devCosts, memsize*disp_range);
    
  /*cuda imagens */
  int *devH_leftIm, *devH_rightIm; //tarefa 1
  cudaMalloc((void **)&devH_leftIm, memsize); //tarefa 1
  cudaMalloc((void **)&devH_rightIm, memsize); //tarefa 1

  /*adicionado*/
  int block_x = (disp_range<=32) ? 4 : 3 ; 
  int block_y = (disp_range<=32) ? 4 : 2; 
  int block_z = disp_range; // 512

  int grid_x = ceil((float)nx/block_x);
  int grid_y = ceil((float)ny/block_y);

  dim3 dimBlock(block_x,block_y,block_z);
  dim3 dimGrid(grid_x,grid_y,1);


  /*copias com cudaMemCopy as imagens para o ponteiro de cuda*/ 
  cudaMemcpy(devH_leftIm, h_leftIm, memsize, cudaMemcpyHostToDevice); //tarefa 1
  cudaMemcpy(devH_rightIm, h_rightIm, memsize, cudaMemcpyHostToDevice); //tarefa 1

  devDetermine_costs<<<dimGrid, dimBlock>>>(devH_leftIm, devH_rightIm, devCosts, nx, ny, disp_range); //t1

  int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
  int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));


  int *devAccumulated_costs;
  int *devDir_accumulated_costs;
  cudaMalloc((void **)&devAccumulated_costs, memsize*disp_range);
  cudaMalloc((void **)&devDir_accumulated_costs, memsize*disp_range);

  int dirx=0,diry=0;
  cudaMemcpy(devAccumulated_costs, accumulated_costs, memsize*disp_range, cudaMemcpyHostToDevice);
  cudaMemcpy(devDir_accumulated_costs, dir_accumulated_costs, memsize*disp_range, cudaMemcpyHostToDevice);

  for(dirx=-1; dirx<2; dirx++) {
    if(dirx==0 && diry==0) continue;
    cudaMemset(devDir_accumulated_costs, 0, nx*ny*disp_range*sizeof(int));
    devIterate_direction( dirx,diry, devH_leftIm, devCosts, devDir_accumulated_costs, nx, ny, disp_range);
    devInplace_sum_views<<<dimGrid, dimBlock>>>(devAccumulated_costs, devDir_accumulated_costs, nx, ny, disp_range);
  }
  dirx=0;
  for(diry=-1; diry<2; diry++) {
    if(dirx==0 && diry==0) continue;
    cudaMemset(devDir_accumulated_costs, 0, nx*ny*disp_range*sizeof(int));
    devIterate_direction( dirx,diry, devH_leftIm, devCosts, devDir_accumulated_costs, nx, ny, disp_range);
    devInplace_sum_views<<<dimGrid, dimBlock>>>(devAccumulated_costs, devDir_accumulated_costs, nx, ny, disp_range);
  }
  cudaMemcpy(accumulated_costs, devAccumulated_costs, memsize*disp_range, cudaMemcpyDeviceToHost);


   /* libertar a memoria do device que ja nao vai ser usada */
  cudaFree(devCosts); //t1
  cudaFree(devH_leftIm); //t1
  cudaFree(devH_rightIm); //t1
  //free(dir_accumulated_costs);
  create_disparity_view( accumulated_costs, h_dispIm, nx, ny, disp_range );  //pq temos que fazer isto? 
  //free(accumulated_costs);
  cudaFree(devDir_accumulated_costs);
  cudaFree(devAccumulated_costs);
}

// print command line format
void usage(char *command) 
{
    printf("Usage: %s [-h] [-d device] [-l leftimage] [-r rightimage] [-o dev_dispimage] [-t host_dispimage] [-p disprange] \n",command);
}

// main
int main( int argc, char** argv) 
{

    // default command line options
    int deviceId = 0;
    int disp_range = 32;
    char *leftIn      =(char *)"lbull.pgm",
         *rightIn     =(char *)"rbull.pgm",
         *fileOut     =(char *)"d_dbull.pgm",
         *referenceOut=(char *)"h_dbull.pgm";

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:l:o:r:t:p:h")) !=-1)
    {
        switch(opt)
        {

            case 'd':  // device
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'l': // left image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                leftIn = strdup(optarg);
                break;
            case 'r': // right image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                rightIn = strdup(optarg);
                break;
            case 'o': // output image (from device) filename 
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 't': // output image (from host) filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                referenceOut = strdup(optarg);
                break;
            case 'p': // disp_range
                if(sscanf(optarg,"%d",&disp_range)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 'h': // help
                usage(argv[0]);
                exit(0);
                break;

        }
    }

    if(optind < argc) {
        fprintf(stderr,"Error in arguments\n");
        usage(argv[0]);
        exit(1);
    }

    // select cuda device
    cutilSafeCall( cudaSetDevice( deviceId ) );
    
    // create events to measure host sgm time and device sgm time
    cudaEvent_t startH, stopH, startD, stopD;
    cudaEventCreate(&startH);
    cudaEventCreate(&stopH);
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);

    // allocate host memory
    int* h_ldata=NULL;
    int* h_rdata=NULL;
    unsigned int h,w;

    //load left pgm
    if (cutLoadPGMi(leftIn, (unsigned int **)&h_ldata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", leftIn);
        exit(1);
    }
    //load right pgm
    if (cutLoadPGMi(rightIn, (unsigned int **)&h_rdata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", rightIn);
        exit(1);
    }

    // allocate mem for the result on host side
    int* h_odata = (int*) malloc( h*w*sizeof(int));
    int* reference = (int*) malloc( h*w*sizeof(int));
 
    // sgm at host
    cudaEventRecord( startH, 0 );
    sgmHost(h_ldata, h_rdata, reference, w, h, disp_range);   
    cudaEventRecord( stopH, 0 ); 
    cudaEventSynchronize( stopH );

    // sgm at GPU
    cudaEventRecord( startD, 0 );
    sgmDevice(h_ldata, h_rdata, h_odata, w, h, disp_range);   
    cudaEventRecord( stopD, 0 ); 
    cudaEventSynchronize( stopD );
    
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    float timeH, timeD;
    cudaEventElapsedTime( &timeH, startH, stopH );
    printf( "Host processing time: %f (ms)\n", timeH);
    cudaEventElapsedTime( &timeD, startD, stopD );
    printf( "Device processing time: %f (ms)\n", timeD);

    // save output images
    if (cutSavePGMi(referenceOut, (unsigned int *)reference, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (cutSavePGMi(fileOut,(unsigned int *) h_odata, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    cutFree( h_ldata);
    cutFree( h_rdata);
    free( h_odata);
    free( reference);

    cutilDeviceReset();
}
