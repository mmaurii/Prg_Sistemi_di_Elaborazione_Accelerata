/*
    TIPI:
        __m128i int
            16 epi8, epu8
            8 epi16, epu16
            4 epi32, epu32
            2 epi64

        __m128 float
            4 ps (32 bit)
        
        __m128d double
            2 pd (64 bit)
*/

// Pragma da usare su OneComputer
#pragma GCC target("sse4.2")
#include <immintrin.h>

// Definizione allineamento
#define SSE_DATA_LANE 16
#define VECTOR_LENGTH 32
#define DATA_SIZE 8

int main(void){

    // LETTURA MEMORIA
    // Memoria -> Registri
    // DEVONO ESSERE ALLINEATI A 16 BYTE
    // Load interi
    __m128i _mm_load_si128(__m128i const* mem_addr);
    // Load float
    __m128 _mm_load_ps(float const* mem_addr);
    __m128d _mm_load_pd(double const* mem_addr);

    // SCRITTURA MEMORIA
    // Registri -> Memoria
    void _mm_store_si128 (__m128i* mem_addr, __m128i a);
    void _mm_store_ps (float* mem_addr, __m128 a);
    void _mm_store_pd (double* mem_addr, __m128d a);

    // DICHIARAZIONE: Array allineato a un indirizzo multiplo di 16
    int A[VECTOR_LENGTH] __attribute__((aligned(SSE_DATA_LANE)));

    // ALLOCAZIONE allienata
    void *_mm_malloc(size_t size, size_t align);
    void _mm_free(void* mem_addr);

    // DICHIARAZIONE registro
    __m128i XMM_SSE_REG;
    __m128i *p_A = (__m128i*) A;
    // Copia 32 byte alla volta
    XMM_SSE_REG = _mm_load_si128(p_A);

    //Nel caso di trasferimenti
    for(int i = 0; i < VECTOR_LENGTH * DATA_SIZE / SSE_DATA_LANE; i++){};

    /*
        Compilazione:
        g++ -msse4 main.cpp -o main
    */

    // Performance counter, indica il timestamp
    u_int64_t __rdtsc();

    // Si può estrarre un valore da un registro esteso        
    // Estrazione di 32 bit, con indice imm8
    // C'è anche per 8 e 16 bit
    // L'ordine di memorizzazione: i3 i2 i1 i0
    int _mm_extract_epi32(__m128i a, const int imm8);

    // Operazioni logiche bit a bit
    __m128i _mm_and_si128(__m128i a, __m128i b);
    __m128i _mm_or_si128(__m128i a, __m128i b);
    __m128i _mm_xor_si128(__m128i a, __m128i b);
    // NOT(a) AND b
    __m128i _mm_andnot_si128(__m128i a, __m128i b);
    // Shift a sinistra, ogni 16 bit, immediato
    // Esiste per tutte le dimensioni
    __m128i _mm_slli_epi16 (__m128i a, int imm8);
    // Idem, ma con il registro
    __m128i _mm_sll_epi16 (__m128i a, __m128i count);

    // Shuffle usando una maschera che indica la destinazione finale
    __m128i _mm_shuffle_epi32 (__m128i a, int imm8);

    // Comprime il contenuto di due registri in uno usando i limiti di soglia (unsigned e signed)
    __m128i _mm_packus_epi16 (__m128i a, __m128i b);
    __m128i _mm_packus_epi32 (__m128i a, __m128i b);
    __m128i _mm_packs_epi16 (__m128i a, __m128i b);
    __m128i _mm_packs_epi32 (__m128i a, __m128i b);

    // CVT: passaggio a dimensioni più grandi (ultimi 8 byte)
    __m128i _mm_cvtepi8_epi16 (__m128i a);

    // Blending: combinare il contenuto di due registri con una maschera
    __m128i _mm_blend_epi16 (__m128i a, __m128i b, const int imm8);

    // Inserimento di un valore in una data posizione
    __m128i _mm_insert_epi16 (__m128i a, int i, const int imm8);

    // Istruzioni di confronto che generano delle maschere che si prestano al confronto logico, sempre a cmp b
    __m128i _mm_cmpeq_epi8 (__m128i a, __m128i b);
    __m128i _mm_cmpgt_epi8 (__m128i a, __m128i b);
    __m128i _mm_cmplt_epi8 (__m128i a, __m128i b);

    // Ricerca di massimi e minimi
    __m128i _mm_max_epi32 (__m128i a, __m128i b);
    __m128i _mm_max_epu32 (__m128i a, __m128i b);

    // Addizione (unsigned, signed, con saturazione)
    __m128i _mm_add_epi8 (__m128i a, __m128i b);
    __m128i _mm_add_epi8 (__m128i a, __m128i b);
    __m128i _mm_adds_epi8 (__m128i a, __m128i b);

    // Valore assoluto
    __m128i _mm_abs_epi32 (__m128i a);

    // Addizioni orizzontali
    __m128i _mm_hadd_epi16 (__m128i a, __m128i b);

    // Media approssimata
    __m128i _mm_avg_epu16 (__m128i a, __m128i b);

    // Cambio di segno condizionato
    __m128i _mm_sign_epi16 (__m128i a, __m128i b);

    // Moltiplicazione tra interi (dimensione doppia)
    __m128i _mm_mul_epu32 (__m128i a, __m128i b);

    // Moltiplicazione tra interi (dimensione invariata, tronca la parte più significativa)
    __m128i _mm_mullo_epi16 (__m128i a, __m128i b);
    // Questa tronca quella meno significativa
    __m128i _mm_mulhi_epu16 (__m128i a, __m128i b);

    return 0;
}