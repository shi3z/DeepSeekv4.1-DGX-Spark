#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
// O_DIRECT random reads of given size from a file
int main(int argc,char**argv){
  const char*path=argv[1]; size_t sz=atol(argv[2]); int n=atoi(argv[3]);
  int fd=open(path,O_RDONLY|O_DIRECT); if(fd<0){perror("open");return 1;}
  off_t fsz=lseek(fd,0,SEEK_END);
  void*buf; if(posix_memalign(&buf,4096,sz)){perror("memalign");return 1;}
  struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
  size_t tot=0;
  for(int i=0;i<n;i++){
    off_t off=((off_t)(random()%((fsz-sz)/4096)))*4096;
    ssize_t r=pread(fd,buf,sz,off); if(r<0){perror("pread");return 1;} tot+=r;
  }
  clock_gettime(CLOCK_MONOTONIC,&b);
  double el=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
  printf("O_DIRECT size=%.2f MB n=%d : %.3f s -> %.2f GB/s (%.1f us/read)\n", sz/1e6, n, el, tot/1e9/el, el/n*1e6);
  return 0;
}
