#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <immintrin.h>

#define ROWS 16
#define COLS 16

#define NO_INLINE __attribute__((noinline))
#define ALIGNED16 __attribute__((aligned(16)))


static void transpose(uint8_t *array, int m, int n)
{
    uint8_t *temp=malloc(m*n*sizeof(int));     //need to create a temporary array. 
    memcpy(temp,array,m*n*sizeof(int));
    int i, j;

    for (i = 0; i < m; ++i) 
    {
        for (j = 0; j < n; ++j)
        {
            array[j*m+i]=temp[i*n+j];
        }
    }

    free(temp);
}
void print128_8(__m128i var)
{
	uint8_t val[16];
	memcpy(val, &var, sizeof(val));
	printf(" \n %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u \n ", val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
	return;
}

static inline uint8_t avg(uint8_t a, uint8_t b)
{
	return (uint8_t)(((uint16_t)a + (uint16_t)b + 1) / 2);
}

static inline void haar_x_scalar(uint8_t *output, const uint8_t *input)
{
	for (size_t y = 0; y < ROWS; y++)
	{
		uint8_t tmp_input_row[COLS];
		memcpy(tmp_input_row, &input[y * COLS], COLS);

		for (size_t lim = COLS; lim > 1; lim /= 2)
		{
			for (size_t x = 0; x < lim; x += 2)
			{
				uint8_t a = tmp_input_row[x];
				uint8_t b = tmp_input_row[x + 1];
				uint8_t s = avg(a, b);
				uint8_t d = avg(a, -b);
				tmp_input_row[x / 2] = s;
				output[y * COLS + (lim + x) / 2] = d;
			}
		}

		output[y * COLS] = tmp_input_row[0];
	}
}

static inline void haar_y_scalar(uint8_t *output, const uint8_t *input)
{
	for (size_t x = 0; x < COLS; x++)
	{
		uint8_t tmp_input_col[ROWS];
		for (size_t y = 0; y < ROWS; y++)
		{
			tmp_input_col[y] = input[y * COLS + x];
		}

		for (size_t lim = COLS; lim > 1; lim /= 2)
		{
			for (size_t y = 0; y < lim; y += 2)
			{
				uint8_t a = tmp_input_col[y];
				uint8_t b = tmp_input_col[y + 1];
				uint8_t s = avg(a, b);
				uint8_t d = avg(a, -b);
				tmp_input_col[y / 2] = s;
				output[(lim + y) / 2 * COLS + x] = d;
			}
		}

		output[x] = tmp_input_col[0];
	}
}

NO_INLINE static void haar_scalar(uint8_t *output, const uint8_t *input)
{
	uint8_t tmp[ROWS * COLS];
	haar_x_scalar(tmp, input);
	haar_y_scalar(output, tmp);
}

static inline void haar_y_simd(uint8_t *output, const uint8_t *input)
{

	uint8_t temp[COLS*ROWS];
	memcpy(temp, &input[0], ROWS*COLS);

	transpose(temp,ROWS,COLS);

	__m128i *out_vec = (__m128i *)output;
	__m128i tmp_input_row[COLS];

	__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

	__m128i control = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1);
	__m128i control1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);

	__m128i mask1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0);
	__m128i mask2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0);
	__m128i mask4 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0);
	__m128i mask8 = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0);

	__m128i shuffle_mask1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	__m128i shuffle_mask2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
	__m128i shuffle_mask4 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0);
	__m128i shuffle_mask8 = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	for (int y = 0; y < 16; y += 1) // adjust ROWS
	{
		memcpy(tmp_input_row, &temp[y * COLS], COLS);
		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1);	 // xmm1 takes the odd values of the 16 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);		 // xmm2 takes the even values of the 16 ints
		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the values from 0 -> 7 are correct
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm3 = _mm_avg_epu8(xmm1, xmm2); // take only from byte 8 -> 15

		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1); // xmm1 takes the odd values of the 4 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);	 // xmm2 takes the even values of the 4 ints

		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the values from 0 -> 3 are correct the wrong ones does not matter
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm4 = _mm_avg_epu8(xmm1, xmm2); // take only from byte 4 -> 7

		//////////
		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1); // xmm1 takes the odd values of the 4 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);	 // xmm2 takes the even values of the 4 ints

		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the values from 0 -> 1 are correct the wrong ones does not matter
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm5 = _mm_avg_epu8(xmm1, xmm2); // take only from byte 2 -> 3

		//////////
		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1); // xmm1 takes the odd values of the 4 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);	 // xmm2 takes the even values of the 4 ints

		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the first element 0 and put it in the first element of output
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm6 = _mm_avg_epu8(xmm1, xmm2); // take only byte 1

		///// 	Putting the values in the right position to store them

		__m128i mask00 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255);

		xmm6 = _mm_shuffle_epi8(xmm6, shuffle_mask1);
		xmm5 = _mm_shuffle_epi8(xmm5, shuffle_mask2);
		xmm4 = _mm_shuffle_epi8(xmm4, shuffle_mask4);
		xmm3 = _mm_shuffle_epi8(xmm3, shuffle_mask8);

		xmm6 = _mm_and_si128(xmm6, mask1);
		xmm5 = _mm_and_si128(xmm5, mask2);
		xmm4 = _mm_and_si128(xmm4, mask4);
		xmm3 = _mm_and_si128(xmm3, mask8);

		xmm7 = _mm_and_si128(tmp_input_row[0], mask00);
		xmm7 = _mm_or_si128(xmm7, xmm6);

		xmm7 = _mm_or_si128(xmm7, xmm5);
		xmm7 = _mm_or_si128(xmm7, xmm4);
		xmm7 = _mm_or_si128(xmm7, xmm3);

		_mm_storeu_si128(&out_vec[y], xmm7); // initialiying to zeros
	}

	memcpy(output, out_vec, ROWS * COLS);
	transpose(output,ROWS,COLS);

	

}

static inline void haar_x_simd(uint8_t *output, const uint8_t *input)
{

	__m128i *out_vec = (__m128i *)output;
	__m128i tmp_input_row[COLS];

	__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

	__m128i control = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1);
	__m128i control1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);

	__m128i mask1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0);
	__m128i mask2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0);
	__m128i mask4 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0);
	__m128i mask8 = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0);

	__m128i shuffle_mask1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	__m128i shuffle_mask2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
	__m128i shuffle_mask4 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0);
	__m128i shuffle_mask8 = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	for (int y = 0; y < 16; y += 1) // adjust ROWS
	{
		memcpy(tmp_input_row, &input[y * COLS], COLS);
		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1);	 // xmm1 takes the odd values of the 16 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);		 // xmm2 takes the even values of the 16 ints
		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the values from 0 -> 7 are correct
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm3 = _mm_avg_epu8(xmm1, xmm2); // take only from byte 8 -> 15

		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1); // xmm1 takes the odd values of the 4 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);	 // xmm2 takes the even values of the 4 ints

		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the values from 0 -> 3 are correct the wrong ones does not matter
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm4 = _mm_avg_epu8(xmm1, xmm2); // take only from byte 4 -> 7

		//////////
		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1); // xmm1 takes the odd values of the 4 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);	 // xmm2 takes the even values of the 4 ints

		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the values from 0 -> 1 are correct the wrong ones does not matter
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm5 = _mm_avg_epu8(xmm1, xmm2); // take only from byte 2 -> 3

		//////////
		xmm0 = _mm_load_si128(&tmp_input_row[0]);
		xmm1 = _mm_shuffle_epi8(xmm0, control1); // xmm1 takes the odd values of the 4 ints
		xmm2 = _mm_shuffle_epi8(xmm0, control);	 // xmm2 takes the even values of the 4 ints

		tmp_input_row[0] = _mm_avg_epu8(xmm1, xmm2); // take only the first element 0 and put it in the first element of output
		xmm2 = _mm_sub_epi8(shuffle_mask1, xmm2);
		xmm6 = _mm_avg_epu8(xmm1, xmm2); // take only byte 1

		///// 	Putting the values in the right position to store them

		__m128i mask00 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255);

		xmm6 = _mm_shuffle_epi8(xmm6, shuffle_mask1);
		xmm5 = _mm_shuffle_epi8(xmm5, shuffle_mask2);
		xmm4 = _mm_shuffle_epi8(xmm4, shuffle_mask4);
		xmm3 = _mm_shuffle_epi8(xmm3, shuffle_mask8);

		xmm6 = _mm_and_si128(xmm6, mask1);
		xmm5 = _mm_and_si128(xmm5, mask2);
		xmm4 = _mm_and_si128(xmm4, mask4);
		xmm3 = _mm_and_si128(xmm3, mask8);

		xmm7 = _mm_and_si128(tmp_input_row[0], mask00);
		xmm7 = _mm_or_si128(xmm7, xmm6);

		xmm7 = _mm_or_si128(xmm7, xmm5);
		xmm7 = _mm_or_si128(xmm7, xmm4);
		xmm7 = _mm_or_si128(xmm7, xmm3);

		_mm_storeu_si128(&out_vec[y], xmm7); // initialiying to zeros
	}

	memcpy(output, out_vec, ROWS * COLS);

	// haar_x_scalar(output, input);
}

// haar_y_scalar(output, input);

NO_INLINE static void haar_simd(uint8_t *output, const uint8_t *input)
{
	uint8_t tmp[ROWS * COLS] ALIGNED16;
	haar_x_simd(tmp, input);
	haar_y_simd(output, tmp);
}

static int64_t time_diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if (end.tv_nsec - start.tv_nsec < 0)
	{
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	}
	else
	{
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp.tv_sec * 1000000000 + temp.tv_nsec;
}

static void benchmark(
	void (*fn)(uint8_t *, const uint8_t *),
	uint8_t *output, const uint8_t *input, size_t iterations, const char *msg)
{
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	for (size_t i = 0; i < iterations; i++)
	{
		fn(output, input);
	}

	clock_gettime(CLOCK_REALTIME, &end);
	double avg = (double)time_diff(start, end) / iterations;
	printf("%10s:\t %.3f ns\n", msg, avg);
}

static uint8_t *alloc_matrix()
{
	return memalign(16, ROWS * COLS);
}

static void init_matrix(uint8_t *matrix)
{
	for (size_t y = 0; y < ROWS; y++)
	{
		for (size_t x = 0; x < COLS; x++)
		{
			matrix[y * COLS + x] = (uint8_t)(rand() & UINT8_MAX);
		}
	}
}

static bool compare_matrix(uint8_t *expected, uint8_t *actual)
{
	bool correct = true;
	for (size_t y = 0; y < ROWS; y++)
	{
		for (size_t x = 0; x < COLS; x++)
		{
			uint8_t e = expected[y * COLS + x];
			uint8_t a = actual[y * COLS + x];
			if (e != a)
			{
				printf(
					"Failed at (y=%zu, x=%zu): expected=%u, actual=%u\n",
					y, x, e, a);
				correct = false;
			}
		}
	}
	return correct;
}

int main()
{
	uint8_t *input = alloc_matrix();
	uint8_t *output_scalar = alloc_matrix();
	uint8_t *output_simd = alloc_matrix();

	/* Check for correctness */
	for (size_t n = 0; n < 100; n++)
	{
		init_matrix(input);
		haar_scalar(output_scalar, input);
		haar_simd(output_simd, input);
		if (!compare_matrix(output_scalar, output_simd))
		{
			break;
		}
	}

	/* Benchmark */
	init_matrix(input);
	benchmark(haar_scalar, output_scalar, input, 3000000, "scalar");
	benchmark(haar_simd, output_simd, input, 3000000, "simd");
	benchmark(haar_x_scalar, output_scalar, input, 3000000, "scalar_x");
	benchmark(haar_x_simd, output_simd, input, 3000000, "simd_x");
	benchmark(haar_y_scalar, output_scalar, input, 3000000, "scalar_y");
	benchmark(haar_y_simd, output_simd, input, 3000000, "simd_y");

	return 1;
}
