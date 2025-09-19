/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    lora_app.h
  * @author  MCD Application Team
  * @brief   Header of application of the LRWAN Middleware
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __LORA_APP_H__
#define __LORA_APP_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/

/* LoraWAN application configuration (Mw is configured by lorawan_conf.h) */
#define ACTIVE_REGION                               LORAMAC_REGION_AU915

/*!
 * CAYENNE_LPP is myDevices Application server.
 */
/*#define CAYENNE_LPP*/

/*!
 * Defines the application data transmission duty cycle. 10s, value in [ms].
 */
#define APP_TX_DUTYCYCLE                            10000

/*!
 * LoRaWAN User application port
 * @note do not use 224. It is reserved for certification
 */
#define LORAWAN_USER_APP_PORT                       2

/*!
 * LoRaWAN Switch class application port
 * @note do not use 224. It is reserved for certification
 */
#define LORAWAN_SWITCH_CLASS_PORT                   3

/*!
 * LoRaWAN default endNode class port
 */
#define LORAWAN_DEFAULT_CLASS                       CLASS_A

/*!
 * LoRaWAN default confirm state
 */
#define LORAWAN_DEFAULT_CONFIRMED_MSG_STATE         LORAMAC_HANDLER_UNCONFIRMED_MSG

/*!
 * LoRaWAN Adaptive Data Rate
 * @note Please note that when ADR is enabled the end-device should be static
 */
#define LORAWAN_ADR_STATE                           LORAMAC_HANDLER_ADR_OFF//on

/*!
 * LoRaWAN Default data Rate Data Rate
 * @note Please note that LORAWAN_DEFAULT_DATA_RATE is used only when LORAWAN_ADR_STATE is disabled
 */
#define LORAWAN_DEFAULT_DATA_RATE                   DR_0

/*!
 * LoRaWAN default activation type
 */
#define LORAWAN_DEFAULT_ACTIVATION_TYPE             ACTIVATION_TYPE_OTAA//otaa

/*!
 * User application data buffer size
 */
#define LORAWAN_APP_DATA_BUFFER_MAX_SIZE            242

/*!
 * Default Unicast ping slots periodicity
 *
 * \remark periodicity is equal to 2^LORAWAN_DEFAULT_PING_SLOT_PERIODICITY seconds
 *         example: 2^3 = 8 seconds. The end-device will open an Rx slot every 8 seconds.
 */
#define LORAWAN_DEFAULT_PING_SLOT_PERIODICITY       4

/* USER CODE BEGIN EC */
#define PA15_RESERVED_Pin GPIO_PIN_15
#define PA15_RESERVED_GPIO_Port GPIOA
#define PB3_RESERVED_Pin GPIO_PIN_3
#define PB3_RESERVED_GPIO_Port GPIOB
#define PA12_RESERVED_Pin GPIO_PIN_12
#define PA12_RESERVED_GPIO_Port GPIOA
#define PB4_RESERVED_Pin GPIO_PIN_4
#define PB4_RESERVED_GPIO_Port GPIOB
#define PB4_RESERVED_EXTI_IRQn EXTI4_15_IRQn
#define PC13_RESERVED_Pin GPIO_PIN_13
#define PC13_RESERVED_GPIO_Port GPIOC
#define PC13_RESERVED_EXTI_IRQn EXTI4_15_IRQn
#define PC1_RESERVED_Pin GPIO_PIN_1
#define PC1_RESERVED_GPIO_Port GPIOC
#define PC0_RESERVED_Pin GPIO_PIN_0
#define PC0_RESERVED_GPIO_Port GPIOC
#define PB1_RESERVED_Pin GPIO_PIN_1
#define PB1_RESERVED_GPIO_Port GPIOB
#define PB1_RESERVED_EXTI_IRQn EXTI0_1_IRQn
#define PA1_RESERVED_Pin GPIO_PIN_1
#define PA1_RESERVED_GPIO_Port GPIOA
#define PC2_RESERVED_Pin GPIO_PIN_2
#define PC2_RESERVED_GPIO_Port GPIOC
#define PA7_RESERVED_Pin GPIO_PIN_7
#define PA7_RESERVED_GPIO_Port GPIOA
#define STLINK_RX_Pin GPIO_PIN_2
#define STLINK_RX_GPIO_Port GPIOA
#define PB0_RESERVED_Pin GPIO_PIN_0
#define PB0_RESERVED_GPIO_Port GPIOB
#define PB0_RESERVED_EXTI_IRQn EXTI0_1_IRQn
#define PA6_RESERVED_Pin GPIO_PIN_6
#define PA6_RESERVED_GPIO_Port GPIOA
#define STLINK_TX_Pin GPIO_PIN_3
#define STLINK_TX_GPIO_Port GPIOA
/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
/**
  * @brief  Init Lora Application
  */
void LoRaWAN_Init(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

#ifdef __cplusplus
}
#endif

#endif /*__LORA_APP_H__*/

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
