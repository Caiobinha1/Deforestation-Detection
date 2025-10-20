/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_lorawan.h"
#include "RegionAU915.h"
#include "sys_app.h"
#include "dma.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define HAL_MAX_DELAY      0xFFFFFFFFU
#define SPI_TIMEOUT_MS HAL_MAX_DELAY
#define BUFFER_SIZE 1
#define SPI_TRIGGER_VALUE 0x55
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
UART_HandleTypeDef huart2;
SPI_HandleTypeDef hspi2;
uint8_t spi_rx_buffer = 0;
uint8_t spi_tx_buffer = 0;
uint8_t old_value = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_SPI2_Init_Slave(void);
void MX_GPIO_Init(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

int main(void){

	HAL_Init();
	SystemClock_Config();

	/* USER CODE BEGIN 2 */
	MX_DMA_Init();
	MX_SPI2_Init_Slave();

	HAL_SPI_Receive_DMA(&hspi2, &spi_rx_buffer, BUFFER_SIZE);
	//HAL_SPI_Transmit_DMA(&hspi2, &spi_tx_buffer, BUFFER_SIZE);
	// Now initialize LoRaWAN
	MX_LoRaWAN_Init();
	/* USER CODE END 2 */

	for(uint8_t i = 0; i < 82; i++){
		MX_LoRaWAN_Process();
	}

	while(1){

		if(spi_rx_buffer != 0){
			MX_LoRaWAN_Process();
		}
		HAL_Delay(5);

	}
}

void SystemClock_Config(void){
	RCC_OscInitTypeDef RCC_OscInitStruct = {0};
	RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

	RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
	RCC_OscInitStruct.HSEState            = RCC_HSE_OFF;
	RCC_OscInitStruct.HSIState            = RCC_HSI_ON;
	RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
	RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSI;
	RCC_OscInitStruct.PLL.PLLMUL          = RCC_PLLMUL_6;
	RCC_OscInitStruct.PLL.PLLDIV          = RCC_PLLDIV_3;

	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK){
		Error_Handler();
	}

	__HAL_RCC_PWR_CLK_ENABLE();
	__HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
	while (__HAL_PWR_GET_FLAG(PWR_FLAG_VOS) != RESET) {};

	RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK){
		Error_Handler();
	}
}


/* USER CODE BEGIN 4 */
static void MX_SPI2_Init_Slave(void){
	hspi2.Instance = SPI2;
	hspi2.Init.Mode = SPI_MODE_SLAVE;
	hspi2.Init.Direction = SPI_DIRECTION_2LINES;
	hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
	hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
	hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
	hspi2.Init.NSS = SPI_NSS_HARD_INPUT;//HARD_INPUT
	hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
	hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
	hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
	hspi2.Init.CRCPolynomial = 7;

	if (HAL_SPI_Init(&hspi2) != HAL_OK)
	{
	Error_Handler();
	}
}


void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi){
	if (hspi->Instance == SPI2) // Check if the callback is for SPI2
	{
		// Data reception complete.
		// Process the 'spi_rx_buffer' here.
		// For continuous reception, re-initiate the DMA transfer immediately:
		HAL_SPI_Receive_DMA(&hspi2, &spi_rx_buffer, BUFFER_SIZE);
		//HAL_GPIO_TogglePin (GPIOA, GPIO_PIN_10);
		//HAL_SPI_Transmit_DMA(&hspi2, &spi_tx_buffer, BUFFER_SIZE);
		// Example: Toggle an LED to visually indicate data reception
		// (Assuming LD1_GPIO_Port and LD1_Pin are defined and configured as GPIO Output)
		//HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin);
	}
}

void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi){
	if (hspi->Instance == SPI2) // Check if the callback is for SPI2
	{
		// Data transmission complete.
		// For continuous transmission, re-initiate the DMA transfer:
		HAL_SPI_Transmit_DMA(&hspi2, &spi_tx_buffer, BUFFER_SIZE);

	}
}


/* USER CODE END 4 */

void Error_Handler(void){
	__disable_irq();
	while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line){
	while (1) {}
}
#endif
