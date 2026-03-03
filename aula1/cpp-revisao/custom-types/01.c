#include <stdio.h>
#include <stdlib.h>

int main(){
    int arr[] = {1, 2, 3, 4, 5};

    size_t size = sizeof(arr) / sizeof(arr[0]);

    printf("Tamanho do array: %zu\n", size);   
    printf("Tamanho do size_t: %zu bytes\n", sizeof(size_t));
    printf("Tamanho do int: %zu bytes\n", sizeof(int));
}