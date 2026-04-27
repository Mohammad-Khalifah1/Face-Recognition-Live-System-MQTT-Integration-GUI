#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include "mqtt_client.h"
#include "cJSON.h"

// ═══════════════════════════════════════════════════════════════════
//  CONFIGURATION
// ═══════════════════════════════════════════════════════════════════
#define WIFI_SSID        "CYBER"
#define WIFI_PASS        "cyberap2025"
#define MQTT_BROKER_URL  "mqtt://192.168.1.24"

// LEDs
#define LED_GREEN_GPIO   GPIO_NUM_5
#define LED_RED_GPIO     GPIO_NUM_19

// MQTT topics  (prefix is "face-id" to match your pipeline)
#define TOPIC_SENSORS    "face-id/sensors"
#define TOPIC_APPROVED   "face-id/approved"
#define TOPIC_REJECTED   "face-id/rejected"

// ═══════════════════════════════════════════════════════════════════
static const char              *TAG             = "FACE_ID_LED";
static esp_mqtt_client_handle_t mqtt_client     = NULL;
static bool                     mqtt_ready      = false;
static bool                     mqtt_initialized = false;

// ── LED helpers ───────────────────────────────────────────────────

static void leds_all_off(void) {
    gpio_set_level(LED_GREEN_GPIO, 0);
    gpio_set_level(LED_RED_GPIO,   0);
    ESP_LOGI(TAG, "LEDs → ALL OFF");
}

static void leds_approved(void) {
    gpio_set_level(LED_GREEN_GPIO, 1);   // green ON
    gpio_set_level(LED_RED_GPIO,   0);   // red OFF
    vTaskDelay(pdMS_TO_TICKS(2000));  // brief delay to show green before turning it off
    gpio_set_level(LED_GREEN_GPIO, 0);   // green OFF
    gpio_set_level(LED_RED_GPIO,   0);   // red OFF
    ESP_LOGI(TAG, "LEDs → GREEN (approved)");
}

static void leds_rejected(void) {
    gpio_set_level(LED_GREEN_GPIO, 0);   // green OFF
    gpio_set_level(LED_RED_GPIO,   1);   // red ON
    vTaskDelay(pdMS_TO_TICKS(2000));  // brief delay to show green before turning it off
    gpio_set_level(LED_GREEN_GPIO, 0);   // green OFF
    gpio_set_level(LED_RED_GPIO,   0);   // red OFF
    ESP_LOGI(TAG, "LEDs → RED (rejected)");
}

// ── MQTT event handler ────────────────────────────────────────────

static void mqtt_event_handler(void *args, esp_event_base_t base,
                                int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;

    switch (event->event_id) {

        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            mqtt_ready = true;
            // Subscribe to all three face-id topics
            esp_mqtt_client_subscribe(mqtt_client, TOPIC_SENSORS,  1);
            esp_mqtt_client_subscribe(mqtt_client, TOPIC_APPROVED, 1);
            esp_mqtt_client_subscribe(mqtt_client, TOPIC_REJECTED, 1);
            ESP_LOGI(TAG, "Subscribed → %s | %s | %s",
                     TOPIC_SENSORS, TOPIC_APPROVED, TOPIC_REJECTED);
            break;

        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "MQTT disconnected");
            mqtt_ready = false;
            leds_all_off();   // safe state while offline
            break;

        case MQTT_EVENT_DATA: {
            // Safe-copy topic & payload
            char topic[64]  = {0};
            char data[256]  = {0};
            int  tl = event->topic_len < 63  ? event->topic_len : 63;
            int  dl = event->data_len  < 255 ? event->data_len  : 255;
            memcpy(topic, event->topic, tl);
            memcpy(data,  event->data,  dl);

            ESP_LOGI(TAG, "MSG  topic=%s  data=%s", topic, data);

            // ── face-id/sensors ───────────────────────────────────
            // Payload: "off"  → turn everything off
            if (strcmp(topic, TOPIC_SENSORS) == 0) {
                if (strstr(data, "off") != NULL) {
                    leds_all_off();
                }
                break;
            }

            // ── face-id/approved ──────────────────────────────────
            // Payload: {"person": "name", "similarity": 87.5}
            if (strcmp(topic, TOPIC_APPROVED) == 0) {
                cJSON *root = cJSON_Parse(data);
                if (root) {
                    cJSON *name = cJSON_GetObjectItemCaseSensitive(root, "person");
                    cJSON *sim  = cJSON_GetObjectItemCaseSensitive(root, "similarity");
                    if (cJSON_IsString(name) && cJSON_IsNumber(sim))
                        ESP_LOGI(TAG, "Approved: %s  (%.1f%%)",
                                 name->valuestring, sim->valuedouble);
                    cJSON_Delete(root);
                }
                leds_approved();
                break;
            }

            // ── face-id/rejected ──────────────────────────────────
            // Payload: {"person": "unknown"}
            if (strcmp(topic, TOPIC_REJECTED) == 0) {
                leds_rejected();
                break;
            }

            break;
        }

        default: break;
    }
}

// ── MQTT init — called only after Wi-Fi has an IP ────────────────

static void init_mqtt(void) {
    if (mqtt_initialized) return;
    mqtt_initialized = true;

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URL,
        .buffer.size        = 512,
        .buffer.out_size    = 512,
    };

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (!mqtt_client) {
        ESP_LOGE(TAG, "esp_mqtt_client_init failed — out of memory!");
        mqtt_initialized = false;
        return;
    }

    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID,
                                   mqtt_event_handler, NULL);
    esp_mqtt_client_start(mqtt_client);
    ESP_LOGI(TAG, "MQTT client started → %s", MQTT_BROKER_URL);
}

// ── Wi-Fi event handler ───────────────────────────────────────────

static void wifi_event_handler(void *arg, esp_event_base_t base,
                                int32_t event_id, void *event_data)
{
    if (base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi disconnected — reconnecting...");
        mqtt_ready = false;
        esp_wifi_connect();

    } else if (base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&e->ip_info.ip));
        init_mqtt();
    }
}

// ── GPIO init ─────────────────────────────────────────────────────

static void init_gpio(void) {
    gpio_config_t led_cfg = {
        .pin_bit_mask = (1ULL << LED_GREEN_GPIO) | (1ULL << LED_RED_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&led_cfg);
    leds_all_off();   // start with both LEDs off
}

// ── Network init ──────────────────────────────────────────────────

static void init_network(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = { .ssid = WIFI_SSID, .password = WIFI_PASS },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
}

// ── Entry point ───────────────────────────────────────────────────

void app_main(void) {
    ESP_LOGI(TAG, "Starting face-ID LED indicator");
    init_gpio();
    init_network();
    // No extra task needed — everything is driven by MQTT callbacks
}
