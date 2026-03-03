#include <stdio.h>
#include <iostream>
#include <stdlib.h>

// Fast Inverse Square Root ( Quake III Arena )
// Calcular 1/sqrt(x) de forma rápida usando manipulação de bits
// https://en.wikipedia.org/wiki/Fast_inverse_square_root
float Q_rsqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * reinterpret_cast<long*>(&y);              // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );                    // what the fuck?
    y  = * reinterpret_cast<float*>(&i);
    y  = y * ( threehalfs - ( x2 * y * y ) );        // 1st iteration
//  y  = y * ( threehalfs - ( x2 * y * y ) );        // 2nd iteration, this can be removed

    return y;
}

int main(){
    /// 1 Caso: Cast " do dia a dia "
    // static_cast

    double pi = 3.14159;
    int pi_arredondado = static_cast<int>(pi); // pi_arredondado = 3
    printf("Valor de pi arredondado: %d\n", pi_arredondado);

    // Pq não usar (int)pi?
    // R: Porque o static_cast é mais seguro, 
    // ele é mais explícito e pode ajudar a evitar erros
    // de conversão acidental. 
    
    // 2 Caso: Cast Definitivio ( Runtime )
    // dynamic_cast

    /*
    
    Animal* animal = new Gato();
    Gato* gato = dynamic_cast<Gato*>(animal);
    
    Isso é util, pois se vc tivesse diversos inputs
    ele garante que seja gato pra ter a função miar

    if(gato != nullptr){
        gato->miar();
    }
    
    */


    // Caso mais importante!
    // reinterpret_cast

    //int:    00000000 00000000 00000000 00101010  (42)
    //char*:  00101010  ← os MESMOS bytes, lido diferente

    //exemplo:

    float f = 3.14f;

    unsigned char* bytes = reinterpret_cast<unsigned char*>(&f);

    for(int i = 0; i < sizeof(float); ++i){
        std::cout << std::hex << (int)(bytes[i]) << " ";
    };


    // Exemplo real de reinterpret_cast:
    // Fast Inverse Square Root ( Quake III Arena )
    
    // Calcular 1/sqrt(x) de forma rápida usando manipulação de bits
    // https://en.wikipedia.org/wiki/Fast_inverse_square_root
    
    // 1/sqrt(4) = 0.5 
    std::cout << Q_rsqrt(4.0f) << std::endl;
    // vai retornar 0.4999 
    
    return 0;
}