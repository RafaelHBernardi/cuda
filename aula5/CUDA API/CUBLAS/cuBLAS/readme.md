# cuBLAS

- NVIDIA CUDA Basic Linear Algebra Subprograms é uma biblioteca acelerada por GPU para acelerar aplicações de IA e HPC ( alta perf.)

- Ela inclui diversas extensões de API que fornecem implementações compatíveis com o padrão da indústria para operações BLAS e GEMM ( General Matrix Multiplication ), além de suporte a fusões de operações altamente otimizados para GPUs NVIDIA

- Preste atenção ao formato das matrizes (shaping/layout) 

- Referencias sobre o shaping:
- https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
- https://chatgpt.com/share/6a55a567-5b9c-83e9-be59-b8a0d0ff663c
## cuBLASLt

- O cuBLASLt (CUDA BLAS Lightweight) é uma extensão do cuBLAS que fornece uma api mais flexível, para melhorar desempenho em trabalhos específicos como modelos de Deep Learning

- Praticamente todos os tipos de dados e chamadas de API aqui estão relacionados com matmul

- Quando um problema não pode ser resolvido por um único kernel da GPU, o cuBLASLt tenta dividir o problema em vários subproblemas e executa o kernel separadamente em cada um deles.

- Nesse contexto entram formatos de baixa precisão como FP16, FP8, INT8, perde um pouco da precisão mas não importa

- Esses formatos podem aumentar significativamente a velocidade da inferência e do treinamento em modelos de IA

## cuBLAS-Xt

- Bem mais devagar, host + gpu solving
- Multi GPU
- P/ Workloads distribuidos
- Escolha o Xt somente para operações que excedem a memória de 1 GPU


## CUTLASS

- Matmul é a operação mais importante no Deep Learning, mas o cuBLAS, não permite de forma simples fundir várias operações em uma unica execução, porém o CUTLASS permite

- Obs: esse paper não utiliza CUTLASS porém é interessante ver como a fusão de operações pode acelerar o processo: { ver só a fig 1 }
- https://arxiv.org/pdf/2205.14135
