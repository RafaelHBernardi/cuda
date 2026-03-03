#include <stdio.h>

int main(){
    int value = 42;
    int* ptr = &value; 
    int** ptr2 = &ptr;
    int***ptr3 = &ptr2;
    // ptr3 -> ptr2 -> ptr -> value
    printf("Valor de value: %d\n", ***ptr3);
}