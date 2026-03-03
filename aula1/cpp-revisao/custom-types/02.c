#include <stdio.h>

typedef struct {
    float x;
    float y;
} Point;

int main(){
    Point p1 = {3.5, 4.5};
    printf("Ponto p1: (%.2f, %.2f)\n", p1.x, p1.y);    
}

