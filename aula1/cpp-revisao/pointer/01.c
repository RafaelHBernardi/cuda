#include <stdio.h>
// gcc -o (nome) (nomedoarquivo.c)
int main(){

    int x = 10;
    int* ptr = &x;

    printf("Valor de x: %d\n", x);
    // Memory address of x  
    printf("Valor de ptr: %p\n", (void*)ptr);
    printf("Valor apontado por ptr: %d\n", *ptr);
    // * conteudo apontado por ponteiro nesse *ptr 


}