

# Atomico

por "atomico" nos referimos a indivisibilidade

## Operacao atomica em CUDA

Uma operação atomica em CUDA é uma operação particular na memoria que garante que seja concluida inteiramente por uma thread antes que outra possa acessar ou modificar a mesma posição
( isso previne race conditions )

Perde velocidade, mas ganha segurança nos dados

## Operacoes

- atomicAdd(int *adress, int val) - Adiciona val atomicamente a adress e retorna o valor antigo
- atomicSub() Idem...
- atomicExch(int* adress, int val) - Troca o valor em adrass por val e retorna o antigo

tem as operações AND,OR,XOR bit a bit entre os valores tbm via
atomicXor atomicOr, atomicAnd

## Exemplo

"Exemplo" de como funciona:

```
lock(posição_de_memória)

valor_antigo = *posição_de_memória

*posição_de_memória = valor_antigo + incremento

unlock(posição_de_memória)

return valor_antigo
```

### Resultado

Os valores não atomicos variaram de 99 a 100, enquanto os
valores atomicos deu 1.000.000 = 1000 x 1000 ( THREADS x BLOCKS )

Isso por conta das race conditions, uma thread não termina ai outra colide
com o valor que tava nessa, vai criando valores errados que no final fica
muito longe de 1 milhão
Obvio que isso custa no tempo de processamento ( sai mais devagar )