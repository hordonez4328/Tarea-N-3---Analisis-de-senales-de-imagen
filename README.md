# Tarea Nº 3: Análisis de Señales de Imagen y Transmisión Digital

**Asignatura:** Comunicaciones I  
**Autor:** Harold Andrés Ordoñez Góngora  
**Universidad:** Universidad Tecnológica de Pereira (UTP)

---

## 📌 Descripción del Proyecto

Este laboratorio implementa el ciclo completo de **procesamiento digital de imágenes, serialización de datos, transmisión en canal digital en banda base, reconstrucción sin pérdidas y conversión cruzada medio-a-medio (Imagen $\rightarrow$ Audio $\rightarrow$ Imagen)**.

El objetivo principal es demostrar empíricamente que la información digital es independiente del soporte físico de transporte: una misma trama binaria puede representarse como luz en pantalla (matriz de píxeles RGB), pulsos de voltaje en un medio de transmisión (código de línea NRZ-L) o variaciones de presión acústica reproducibles en altavoces (señal de audio PCM en formato WAV), conservando el 100% de la fidelidad de los datos originales.

---

## 🚀 Estructura del Laboratorio (7 Etapas)

```
[ Imagen RGB ] ──► [ Espacios de Color ] ──► [ Digitalización Binaria ]
                                                       │
[ Reconstrucción ] ◄── [ Transmisión NRZ-L ] ◄─────────┘
        │
        ▼
[ Imagen ──► Audio WAV ] ──► [ Audio WAV ──► Imagen Reconstruida ]
```

### 1. Selección y Validación de la Imagen Original
* Carga de imagen a color en formato RGB.
* Validación de resolución mínima requerida ($\ge 400 \times 400$ píxeles).

### 2. Transformación de Espacios de Color
* **Escala de Grises:** Cálculo de luminancia monocromática según el estándar de radiodifusión ITU-R BT.601:
  $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
* **Canal Rojo Aislado (*Color Splash*):** Segmentación cromática en el espacio de color HSV (Hue, Saturation, Value). Se preservan los píxeles genuinamente rojos ($H \le 18^\circ$ o $H \ge 342^\circ$, con saturación $S > 35\%$ y brillo $V > 20\%$) y los píxeles restantes se proyectan en escala de grises.

### 3. Digitalización y Representación Binaria
* Cuantificación PCM espacial a 8 bits por componente (24 bits/píxel en RGB $\rightarrow$ 16.7 millones de colores posibles).
* Cálculo del tamaño total de información generada en bits, bytes, KB y MB.
* Inspección de muestras binarias representativas de los píxeles.

### 4. Simulación de Transmisión Digital
* Empaquetamiento de la trama de comunicaciones con cabecera de sincronización y control (12 bytes con alto, ancho y canales en formato Big-Endian `!III`) y carga útil (*payload*).
* Modulación en banda base mediante código de línea **NRZ-L** (*Non-Return to Zero Level*), graficando los pulsos digitales temporales.

### 5. Reconstrucción y Análisis de Fidelidad
* Desempaquetado de la trama, decodificación de metadatos y reensamble de la matriz tridimensional.
* Cálculo de métricas objetivas de distorsión:
  * **Error Cuadrático Medio (MSE):** $\text{MSE} = 0.000000$
  * **Relación Señal a Ruido Pico (PSNR):** $\text{PSNR} = \infty\text{ dB}$
  * Verificación de transmisión exacta sin pérdidas numéricas.

### 6. Conversión de Imagen a Audio (Modulación PCM Acústica)
* Mapeo directo de cada byte de la trama binaria como una muestra discreta de amplitud instantánea de audio.
* Exportación a archivo de audio estándar **WAV** (PCM de 8 bits sin signo, frecuencia de muestreo $F_s = 44.1\text{ kHz}$, 1 canal mono).
* Representación y visualización de la forma de onda en el dominio del tiempo con zoom a nivel de milisegundos.
* Reproducción sonora de la imagen codificada.

### 7. Conversión de Audio a Imagen (Demodulación y Recuperación)
* Lectura del archivo de audio `.wav`, extracción de la cabecera embebida en las muestras de audio y regeneración exacta de la imagen original.
* Comparación píxel a píxel demostrando coincidencia del 100%.

---

## 📂 Archivos del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `laboratorio_transmision.py` | Script principal en Python que ejecuta el flujo completo y genera las figuras y audios. |
| `app.html` | Interfaz web interactiva (HTML5, Tailwind CSS, Canvas API y Web Audio API) para visualización y experimentación en tiempo real. |
| `imagen_original.png` | Imagen original utilizada en la prueba ($780 \times 440$ px). |
| `etapa2_espacios_de_color.png` | Gráfica comparativa: Original RGB, escala de grises y rojo aislado. |
| `etapa4_transmision_flujo_bits.png` | Gráfica del tren de pulsos modulado en banda base NRZ-L. |
| `etapa5_reconstruccion_imagen.png` | Comparativa lado a lado de la imagen original vs. la imagen reconstruida del flujo digital. |
| `imagen_a_audio.wav` | Archivo de audio generado a partir de la trama de píxeles ($44.1\text{ kHz}$ mono). |
| `etapa6_forma_de_onda_audio.png` | Oscilograma de la señal acústica completa y zoom temporal milimétrico. |
| `etapa7_audio_a_imagen_reconstruida.png` | Gráfica de verificación y reconstrucción desde la señal de audio. |
| `imagen_recuperada_desde_audio.png` | Archivo de imagen final exportado tras decodificar el audio WAV. |

---

## 💻 Requisitos e Instalación

### Requisitos de Python:
* Python 3.8 o superior
* Bibliotecas requeridas:
  ```bash
  pip install numpy matplotlib pillow
  ```

### Ejecución del Script:
```bash
python laboratorio_transmision.py
```
*(Opcional: se puede pasar como parámetro la ruta de cualquier otra imagen JPG o PNG)*:
```bash
python laboratorio_transmision.py ruta/a/tu_imagen.png
```

### Visualizador Web Interactivo:
Basta con abrir el archivo `app.html` directamente en cualquier navegador moderno (Chrome, Edge, Firefox, Brave) o servirlo mediante un servidor local:
```bash
python -m http.server 8000
```
Y navegar a `http://localhost:8000/app.html`.
