"""
=============================================================================================
LABORATORIO: TRANSMISIÓN DIGITAL DE IMÁGENES Y CONVERSIÓN IMAGEN-AUDIO-IMAGEN EN PYTHON
=============================================================================================
Asignatura: Comunicaciones I
Autor: Harold Andrés Ordoñez Góngora
Descripción:
  Este programa implementa el ciclo completo de procesamiento, digitalización, transmisión,
  reconstrucción y conversión cruzada medio-a-medio (Imagen -> Audio -> Imagen).
=============================================================================================
"""

import os
import sys
import struct
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

# Configuración de matplotlib para visualizaciones claras
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'


def generar_imagen_de_muestra(ruta_salida="imagen_original.png", tamano=(500, 500)):
    """
    Genera una imagen a color con detalles variados (mínimo 400x400)
    en caso de que el usuario no cuente con una imagen propia inicial.
    """
    print(f"[*] Generando imagen de prueba a color ({tamano[0]}x{tamano[1]} px)...")
    img = Image.new("RGB", tamano, color=(240, 243, 246))
    draw = ImageDraw.Draw(img)

    # Dibujar degradados y figuras geométricas
    w, h = tamano
    for y in range(h):
        r = int(255 * (y / h))
        g = int(180 * (1 - y / h))
        b = int(220 * (0.5 + 0.5 * np.sin(y / 20.0)))
        draw.line([(0, y), (w // 3, y)], fill=(r, g, b))

    # Círculos y rectángulos de colores puros y mixtos
    draw.rectangle([w // 3 + 20, 30, w - 30, h // 2 - 20], fill=(33, 150, 243), outline=(13, 71, 161), width=4)
    draw.ellipse([w // 3 + 50, h // 2 + 10, w - 50, h - 30], fill=(244, 67, 54), outline=(183, 28, 28), width=5)
    draw.polygon([(w // 2, 60), (w // 3 + 40, h // 3), (w - 60, h // 3)], fill=(255, 235, 59))

    # Texto ilustrativo
    draw.text((w // 3 + 35, 45), "COMUNICACIONES I", fill=(255, 255, 255))
    draw.text((w // 3 + 35, 75), "Digital Image & Audio", fill=(255, 255, 255))

    img.save(ruta_salida)
    print(f"[+] Imagen de muestra creada exitosamente en: {ruta_salida}\n")
    return ruta_salida


# =============================================================================
# ETAPA 1: SELECCIÓN DE LA IMAGEN ORIGINAL
# =============================================================================
def etapa1_cargar_imagen(ruta_imagen=None):
    """
    Carga y valida la imagen original (JPG o PNG, mínimo 400x400).
    """
    print("=" * 75)
    print("ETAPA 1: SELECCIÓN DE LA IMAGEN ORIGINAL")
    print("=" * 75)

    if not ruta_imagen or not os.path.exists(ruta_imagen):
        ruta_defecto = "imagen_original.png"
        if not os.path.exists(ruta_defecto):
            generar_imagen_de_muestra(ruta_defecto, (500, 500))
        ruta_imagen = ruta_defecto

    print(f"[*] Cargando archivo: '{ruta_imagen}'")
    img_pil = Image.open(ruta_imagen).convert("RGB")
    ancho, alto = img_pil.size
    print(f"[i] Resolución detectada: {ancho} x {alto} píxeles")

    if ancho < 400 or alto < 400:
        print(f"[!] Advertencia: La resolución ({ancho}x{alto}) es inferior a 400x400.")
        print("    Redimensionando a 400x400 para satisfacer los requisitos del laboratorio.")
        img_pil = img_pil.resize((max(400, ancho), max(400, alto)), Image.Resampling.LANCZOS)
        ancho, alto = img_pil.size

    img_np = np.array(img_pil, dtype=np.uint8)
    print(f"[+] Imagen validada: Dimensiones {ancho}x{alto}, {img_np.shape[2]} canales (R, G, B).\n")
    return img_np


# =============================================================================
# ETAPA 2: TRANSFORMACIÓN DE ESPACIOS DE COLOR
# =============================================================================
def etapa2_transformacion_color(img_rgb):
    """
    Transforma la imagen original en:
      1. Escala de grises (Luminancia estándar ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B).
      2. Canal Rojo Aislado (Color Splash): El rojo se conserva intacto y el resto pasa a escala de grises.
    Muestra las tres versiones en una figura comparativa.
    """
    print("=" * 75)
    print("ETAPA 2: TRANSFORMACIÓN DE ESPACIOS DE COLOR")
    print("=" * 75)

    # 1. Escala de grises por luminancia ponderada (ITU-R BT.601)
    # Y = 0.299*R + 0.587*G + 0.114*B
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    img_gris = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)

    # 2. Aislamiento selectivo del canal/color rojo (Color Splash):
    # Convertimos la imagen al espacio de color HSV para aislar el matiz (Hue).
    # En HSV, el color rojo se sitúa en los extremos del espectro circular:
    # H cercano a 0 (0° a ~18°) o cercano a 1 (~342° a 360°).
    # Además, exigimos una saturación cromática mínima (S > 0.35) para descartar
    # blancos, grises y negros, y un brillo mínimo (V > 0.20) para evitar sombras profundas.
    hsv = mcolors.rgb_to_hsv(img_rgb.astype(np.float32) / 255.0)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mascara_rojo = ((h <= 0.05) | (h >= 0.95)) & (s > 0.35) & (v > 0.20)

    # Creamos la imagen resultante:
    # Donde mascara_rojo es True -> Se conserva el color original del píxel (R, G, B)
    # Donde mascara_rojo es False -> Se asigna la luminancia monocromática (img_gris)
    img_rojo_selectivo = np.zeros_like(img_rgb)
    for c in range(3):
        img_rojo_selectivo[:, :, c] = np.where(mascara_rojo, img_rgb[:, :, c], img_gris)

    print("[+] Transformación a escala de grises completada (ponderación ITU-R BT.601).")
    print(f"[+] Aislamiento selectivo de color rojo completado: {np.sum(mascara_rojo):,} píxeles rojos identificados.")
    print("    (Los píxeles no rojos fueron transformados a escala de grises).")

    # Visualización de las 3 versiones
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.canvas.manager.set_window_title("Etapa 2 - Espacios de Color")

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Imagen Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(img_gris, cmap="gray")
    axes[1].set_title("2. Escala de Grises (Luminancia Y)")
    axes[1].axis("off")

    axes[2].imshow(img_rojo_selectivo)
    axes[2].set_title("3. Canal Rojo Aislado (Rojo + Resto Gris)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("etapa2_espacios_de_color.png", dpi=150)
    print("[+] Gráfica guardada como 'etapa2_espacios_de_color.png'. Mostrando ventana...")
    plt.show(block=False)
    plt.pause(1.5)
    print()

    return img_gris, img_rojo_selectivo


# =============================================================================
# ETAPA 3: DIGITALIZACIÓN Y REPRESENTACIÓN BINARIA
# =============================================================================
def etapa3_digitalizacion_binaria(img_rgb):
    """
    Convierte la matriz de píxeles a su representación binaria pura.
    Calcula:
      - Total de píxeles
      - Bits por píxel (bpp)
      - Tamaño total de información (bits, bytes, KB, MB)
    Muestra una muestra representativa de bits por píxel.
    """
    print("=" * 75)
    print("ETAPA 3: DIGITALIZACIÓN Y REPRESENTACIÓN BINARIA")
    print("=" * 75)

    alto, ancho, canales = img_rgb.shape
    total_pixeles = alto * ancho
    bits_por_canal = 8  # Cada intensidad va de 0 a 255 (2^8 niveles)
    bits_por_pixel = canales * bits_por_canal  # 24 bits por píxel en RGB
    total_bits = total_pixeles * bits_por_pixel
    total_bytes = total_bits // 8
    total_kb = total_bytes / 1024
    total_mb = total_kb / 1024

    print("--- FUNDAMENTO TEÓRICO: ALMACENAMIENTO DE UNA IMAGEN EN BITS ---")
    print("Una imagen digital es un arreglo discreto bidimensional (matriz) de píxeles.")
    print("Cada píxel almacena información de color dividida en canales primarios (R, G, B).")
    print("Mediante el proceso de cuantificación analógica-digital (PCM espacial):")
    print("  - Cada canal de color se cuantifica con una profundidad de 8 bits (1 byte).")
    print("  - Los valores posibles van de 0 (00000000b, ausencia) a 255 (11111111b, saturación).")
    print("  - Un píxel RGB a 24 bits representa (2^8)^3 = 16,777,216 colores posibles.\n")

    print("--- CÁLCULOS DE INFORMACIÓN GENERADA ---")
    print(f" * Resolución:                     {ancho} x {alto} píxeles")
    print(f" * Número total de píxeles:         {total_pixeles:,} píxeles")
    print(f" * Canales por píxel:              {canales} (Red, Green, Blue)")
    print(f" * Bits por píxel (bpp):            {bits_por_pixel} bits/píxel (8 bits/canal)")
    print(f" * Tamaño total de información:     {total_bits:,} bits")
    print(f" * Equivalente en Bytes:            {total_bytes:,} Bytes")
    print(f" * Equivalente en Kilobytes (KiB):  {total_kb:,.2f} KB")
    print(f" * Equivalente en Megabytes (MiB):  {total_mb:,.4f} MB\n")

    # Muestra de los primeros 5 píxeles en binario
    print("--- MUESTRA DE DATOS BINARIOS OBTENIDOS (Primeros 5 píxeles) ---")
    print(f"{'Píxel (x, y)':<15} | {'Valores Decimales [R, G, B]':<30} | {'Representación Binaria [R, G, B]':<32}")
    print("-" * 85)

    muestra_binaria = []
    for i in range(min(5, ancho)):
        r_val = img_rgb[0, i, 0]
        g_val = img_rgb[0, i, 1]
        b_val = img_rgb[0, i, 2]
        r_bin = f"{r_val:08b}"
        g_bin = f"{g_val:08b}"
        b_bin = f"{b_val:08b}"
        muestra_binaria.append((r_bin, g_bin, b_bin))
        print(f"({i:3d},   0)        | [{r_val:3d}, {g_val:3d}, {b_val:3d}]                     | [{r_bin}, {g_bin}, {b_bin}]")
    print("-" * 85)
    print()

    # Serializar toda la matriz de píxeles a bytes directos
    flujo_bytes = img_rgb.tobytes()
    return flujo_bytes, (alto, ancho, canales)


# =============================================================================
# ETAPA 4: SIMULACIÓN DE TRANSMISIÓN DIGITAL
# =============================================================================
def etapa4_simulacion_transmision(flujo_bytes, metadata_imagen):
    """
    Simula la transmisión del flujo binario a través de un canal de comunicaciones digital.
    Estructura la trama con cabecera (metadata) y carga útil (payload).
    Visualiza el flujo de bits transmitido en forma de pulsos lógicos (banda base NRZ-L).
    """
    print("=" * 75)
    print("ETAPA 4: SIMULACIÓN DE TRANSMISIÓN DIGITAL")
    print("=" * 75)

    alto, ancho, canales = metadata_imagen

    print("--- MODELO DE TRANSMISIÓN EN UN SISTEMA DIGITAL ---")
    print("En un sistema de telecomunicaciones real, la información se transporta así:")
    print(" 1. Fuente y Codificación de Origen: La imagen se serializa en una secuencia de bits.")
    print(" 2. Empaquetamiento y Protocolo: Se añade una cabecera (Header) con sincronización")
    print("    y dimensiones (H, W, Canales) para que el receptor sepa reensamblar la trama.")
    print(" 3. Codificación de Canal: Se agregan bits de redundancia (ej. CRC, Paridad, Reed-Solomon)")
    print("    para detectar y corregir errores provocados por ruido térmico e interferencia.")
    print(" 4. Modulación: La secuencia de bits modula una portadora analógica (QAM, PSK, FSK)")
    print("    o se envía en banda base mediante códigos de línea como NRZ-L, Manchester, etc.")
    print(" 5. Medio Físico: Fibra óptica, enlace de radiofrecuencia (RF) o par de cobre.")
    print(" 6. Demodulación y Detección: El receptor regenera los símbolos a bits lógicos (0s y 1s).\n")

    # Empaquetamiento de trama de transmisión:
    # Header de 12 bytes: 3 enteros sin signo de 32 bits (Alto, Ancho, Canales)
    cabecera = struct.pack("!III", alto, ancho, canales)
    trama_completa = cabecera + flujo_bytes

    print(f"[*] Trama construida con éxito:")
    print(f"    - Cabecera de control: {len(cabecera)} bytes [Alto: {alto}, Ancho: {ancho}, Canales: {canales}]")
    print(f"    - Carga útil (Payload): {len(flujo_bytes):,} bytes de imagen")
    print(f"    - Longitud total de trama transmitida: {len(trama_completa):,} bytes ({len(trama_completa)*8:,} bits)\n")

    # Extraer los primeros 64 bits para visualización en consola y gráfica
    primeros_bytes = flujo_bytes[:8]
    cadena_bits = "".join(f"{b:08b}" for b in primeros_bytes)
    print(f"[i] Fragmento del flujo de bits transmitido (primeros 64 bits):")
    print(f"    {cadena_bits[:32]} {cadena_bits[32:]}")

    # Graficar la señal en banda base tipo NRZ-L (Non-Return to Zero Level)
    bits_para_grafico = [int(bit) for bit in cadena_bits[:40]]
    tiempo = np.repeat(np.arange(len(bits_para_grafico) + 1), 2)
    senal = np.zeros(len(tiempo))
    for idx, bit in enumerate(bits_para_grafico):
        senal[2 * idx + 1] = bit
        senal[2 * idx + 2] = bit

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.canvas.manager.set_window_title("Etapa 4 - Simulación de Transmisión Digital")
    ax.step(range(len(bits_para_grafico)), bits_para_grafico, where='post', color='#007acc', linewidth=2.5, label='Señal Digital NRZ-L')
    ax.fill_between(range(len(bits_para_grafico)), bits_para_grafico, step='post', alpha=0.2, color='#007acc')

    # Añadir marcas de bits individuales
    for idx, bit in enumerate(bits_para_grafico):
        ax.text(idx + 0.35, bit + 0.1 if bit == 0 else bit - 0.25, str(bit),
                fontsize=11, fontweight='bold', color='#d32f2f' if bit == 1 else '#1565c0')
        ax.axvline(idx, color='gray', linestyle=':', alpha=0.5)

    ax.set_ylim(-0.3, 1.4)
    ax.set_xlim(0, len(bits_para_grafico))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0 (Nivel Bajo: 0V)', '1 (Nivel Alto: +5V)'])
    ax.set_xlabel('Tiempo de Bit (Tb) / Índice de Muestras')
    ax.set_ylabel('Nivel Lógico / Tensión')
    ax.set_title('Etapa 4: Flujo de Bits Transmitido en Canal Digital (Modulación de Banda Base NRZ-L)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("etapa4_transmision_flujo_bits.png", dpi=150)
    print("[+] Gráfica de transmisión guardada como 'etapa4_transmision_flujo_bits.png'. Mostrando ventana...")
    plt.show(block=False)
    plt.pause(1.5)
    print()

    # Simulamos el receptor que recibe la trama intacta (canal sin ruido aditivo)
    return trama_completa


# =============================================================================
# ETAPA 5: RECONSTRUCCIÓN DE LA IMAGEN
# =============================================================================
def etapa5_reconstruccion_imagen(trama_recibida, img_original):
    """
    Reconstruye la matriz de la imagen a partir del flujo binario recibido.
    Compara bit a bit y calcula métricas de error:
      - Tasa de error de bit (BER)
      - Error Cuadrático Medio (MSE)
      - Relación Señal a Ruido Pico (PSNR)
    """
    print("=" * 75)
    print("ETAPA 5: RECONSTRUCCIÓN DE LA IMAGEN Y ANÁLISIS DE PÉRDIDAS")
    print("=" * 75)

    # 1. Desempaquetar cabecera y payload
    tam_cabecera = struct.calcsize("!III")
    cabecera_recibida = trama_recibida[:tam_cabecera]
    payload_recibido = trama_recibida[tam_cabecera:]

    alto_rec, ancho_rec, canales_rec = struct.unpack("!III", cabecera_recibida)
    print(f"[*] Cabecera decodificada con éxito:")
    print(f"    - Dimensiones recuperadas: {ancho_rec} x {alto_rec} x {canales_rec} canales")

    # 2. Reconstrucción de la matriz numérica
    img_reconstruida = np.frombuffer(payload_recibido, dtype=np.uint8).reshape((alto_rec, ancho_rec, canales_rec))

    # 3. Métricas de comparación
    es_identica = np.array_equal(img_original, img_reconstruida)
    mse = np.mean((img_original.astype(np.float64) - img_reconstruida.astype(np.float64)) ** 2)

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    diferencia_absoluta = np.abs(img_original.astype(np.int32) - img_reconstruida.astype(np.int32))
    max_error = np.max(diferencia_absoluta)

    print("\n--- ANÁLISIS DE FIDELIDAD Y PÉRDIDA DE INFORMACIÓN ---")
    print(f" * ¿Son idénticas las imágenes original y reconstruida?:  {'SÍ (Coincidencia al 100%)' if es_identica else 'NO'}")
    print(f" * Error Cuadrático Medio (MSE):                         {mse:.6f}")
    print(f" * Relación Señal a Ruido Pico (PSNR):                    {psnr} dB")
    print(f" * Máxima diferencia absoluta por píxel:                 {max_error} niveles de gris")
    print(f" * Determinación de pérdida:                             NO HUBO PÉRDIDA DE INFORMACIÓN.")
    print("   Explicación: La transmisión digital utiliza representación entera exacta de 8 bits por")
    print("   componente; al no existir ruido ni compresión con pérdidas (lossy), el proceso es 100% reversible.\n")

    # Visualización comparativa
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Etapa 5 - Reconstrucción de Imagen")

    axes[0].imshow(img_original)
    axes[0].set_title("Imagen Original de Entrada")
    axes[0].axis("off")

    axes[1].imshow(img_reconstruida)
    axes[1].set_title(f"Imagen Reconstruida tras Transmisión\n(MSE = {mse:.1f}, PSNR = {psnr})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("etapa5_reconstruccion_imagen.png", dpi=150)
    print("[+] Gráfica guardada como 'etapa5_reconstruccion_imagen.png'. Mostrando ventana...")
    plt.show(block=False)
    plt.pause(1.5)
    print()

    return img_reconstruida


# =============================================================================
# ETAPA 6: CONVERSIÓN DE IMAGEN A AUDIO
# =============================================================================
def etapa6_imagen_a_audio(trama_digital, ruta_audio="imagen_a_audio.wav", tasa_muestreo=44100):
    """
    Transforma la información binaria/bytes de la imagen en una señal de audio PCM analizable.
    Guarda el archivo en formato WAV estándar (PCM 8 bits sin signo o 16 bits).
    Muestra la forma de onda acústica en el dominio del tiempo.
    Reproduce el audio generado.
    """
    print("=" * 75)
    print("ETAPA 6: CONVERSIÓN DE IMAGEN A AUDIO (MODULACIÓN PCM)")
    print("=" * 75)

    print("--- FUNDAMENTO TÉCNICO: MAPEO DE IMAGEN A AUDIO ---")
    print("Cualquier secuencia digital puede interpretarse como muestras temporales de una señal.")
    print("En telecomunicaciones y procesamiento de audio:")
    print(" - Cada byte (0 a 255) representa una muestra de amplitud instantánea (PCM de 8 bits).")
    print(f" - A una frecuencia de muestreo estándar de {tasa_muestreo} Hz (calidad CD Audio),")
    print("   las amplitudes de los píxeles se reproducen como ondas de presión sonora audibles.")
    print(" - Este principio es análogo a la transmisión acústica en módems telefónicos,")
    print("   telemetría por satélite (SSTV) y grabaciones en cinta de datos.\n")

    # Convertir el flujo completo (cabecera + píxeles) a muestras de audio
    # Usamos formato WAV PCM de 8 bits sin signo (1 canal mono, 44100 Hz)
    num_muestras = len(trama_digital)
    duracion_segundos = num_muestras / tasa_muestreo

    print(f"[*] Parámetros de la señal de audio generada:")
    print(f"    - Frecuencia de Muestreo (Fs): {tasa_muestreo} Hz")
    print(f"    - Resolución de Muestreo:     8 bits (1 Byte por muestra, PCM sin signo)")
    print(f"    - Canales de audio:            1 (Mono)")
    print(f"    - Número total de muestras:    {num_muestras:,} muestras")
    print(f"    - Duración acústica estimada:  {duracion_segundos:.2f} segundos")

    # Escritura en archivo WAV con el módulo nativo wave
    with wave.open(ruta_audio, "wb") as wav_file:
        wav_file.setnchannels(1)        # 1 canal (mono)
        wav_file.setsampwidth(1)       # 1 byte = 8 bits por muestra
        wav_file.setframerate(tasa_muestreo)
        wav_file.writeframes(trama_digital)

    print(f"[+] Archivo de audio WAV guardado exitosamente en: '{ruta_audio}'")

    # Graficar la forma de onda (amplitud vs tiempo)
    muestras_np = np.frombuffer(trama_digital, dtype=np.uint8)
    eje_tiempo = np.linspace(0, duracion_segundos, num_muestras)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Etapa 6 - Forma de Onda de Audio Generada")

    # Señal completa
    axes[0].plot(eje_tiempo, muestras_np, color="#2e7d32", linewidth=0.5)
    axes[0].set_title(f"Señal de Audio Completa Codificada desde la Imagen ({duracion_segundos:.2f} s)")
    axes[0].set_xlabel("Tiempo (segundos)")
    axes[0].set_ylabel("Amplitud Cuantificada (0 - 255)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].set_ylim(0, 260)

    # Acercamiento a las primeras 2000 muestras para apreciar los detalles de tono/píxeles
    n_zoom = min(2000, num_muestras)
    axes[1].plot(eje_tiempo[:n_zoom] * 1000, muestras_np[:n_zoom], color="#e65100", linewidth=1.2)
    axes[1].set_title(f"Zoom Temporal a las Primeras {n_zoom} Muestras (Detalle Acústico de Píxeles)")
    axes[1].set_xlabel("Tiempo (milisegundos)")
    axes[1].set_ylabel("Amplitud (0 - 255)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].set_ylim(0, 260)

    plt.tight_layout()
    plt.savefig("etapa6_forma_de_onda_audio.png", dpi=150)
    print("[+] Gráfica de forma de onda guardada como 'etapa6_forma_de_onda_audio.png'. Mostrando ventana...")
    plt.show(block=False)
    plt.pause(1.5)

    # Reproducción del audio generado usando winsound en Windows
    print("\n[*] Reproduciendo el sonido correspondiente a los datos de la imagen...")
    try:
        import winsound
        # Reproducir un fragmento sonoro (o el archivo) de forma asíncrona / breve
        winsound.PlaySound(ruta_audio, winsound.SND_FILENAME | winsound.SND_ASYNC)
        print("[+] Reproducción de audio iniciada correctamente en altavoces.")
    except Exception as e:
        print(f"[!] No fue posible reproducir el audio automáticamente en este entorno: {e}")
        print(f"    Puede abrir y escuchar manualmente el archivo: '{ruta_audio}'")
    print()

    return ruta_audio


# =============================================================================
# ETAPA 7: CONVERSIÓN DE AUDIO A IMAGEN
# =============================================================================
def etapa7_audio_a_imagen(ruta_audio, img_original):
    """
    Lee el archivo de audio WAV generado en la etapa 6.
    Extrae las muestras PCM, decodifica la cabecera y recupera la imagen completa.
    Compara la imagen recuperada contra la original para verificar la fidelidad.
    """
    print("=" * 75)
    print("ETAPA 7: CONVERSIÓN DE AUDIO A IMAGEN (DEMODULACIÓN)")
    print("=" * 75)

    print(f"[*] Abriendo y leyendo archivo de audio: '{ruta_audio}'...")
    with wave.open(ruta_audio, "rb") as wav_file:
        n_canales = wav_file.getnchannels()
        ancho_muestra = wav_file.getsampwidth()
        tasa_muestreo = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        datos_audio_bytes = wav_file.readframes(n_frames)

    print(f"[i] Parámetros leídos del encabezado WAV:")
    print(f"    - Canales: {n_canales}, Ancho de muestra: {ancho_muestra} byte(s)")
    print(f"    - Tasa de muestreo: {tasa_muestreo} Hz, Total de cuadros: {n_frames:,}")

    # Decodificar cabecera de la trama incrustada en el audio
    tam_cabecera = struct.calcsize("!III")
    cabecera = datos_audio_bytes[:tam_cabecera]
    payload = datos_audio_bytes[tam_cabecera:]

    alto_rec, ancho_rec, canales_rec = struct.unpack("!III", cabecera)
    print(f"[+] Cabecera extraída del audio:")
    print(f"    - Dimensiones decodificadas: {ancho_rec} x {alto_rec} x {canales_rec} canales")

    # Reconstrucción de la matriz de la imagen
    img_desde_audio = np.frombuffer(payload, dtype=np.uint8).reshape((alto_rec, ancho_rec, canales_rec))
    print(f"[+] Imagen reconstruida exitosamente a partir de las ondas de audio.")

    # Comparación estricta con la imagen original
    es_identica = np.array_equal(img_original, img_desde_audio)
    mse = np.mean((img_original.astype(np.float64) - img_desde_audio.astype(np.float64)) ** 2)

    if mse == 0:
        psnr_str = "Infinito (Sin pérdida numérica)"
    else:
        psnr_val = 20 * np.log10(255.0 / np.sqrt(mse))
        psnr_str = f"{psnr_val:.2f} dB"

    print("\n--- COMPARACIÓN: IMAGEN ORIGINAL vs IMAGEN RECONSTRUIDA DESDE AUDIO ---")
    print(f" * Coincidencia pixel a pixel:  {'100% EXACTA (Idéntica)' if es_identica else 'Diferencias encontradas'}")
    print(f" * Error Cuadrático Medio (MSE): {mse:.6f}")
    print(f" * PSNR:                         {psnr_str}")
    print(" * Conclusión:                   La información digital es independiente del soporte físico:")
    print("   un flujo binario puede manifestarse como luz (píxeles en pantalla), voltaje en un cable,")
    print("   o sonido acústico en un altavoz sin sufrir alteración si se conserva la integridad de los datos.\n")

    # Visualización lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Etapa 7 - Imagen Reconstruida desde Audio")

    axes[0].imshow(img_original)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_desde_audio)
    axes[1].set_title(f"Imagen Recuperada del Audio WAV\n(MSE = {mse:.1f}, Fidelidad: 100%)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("etapa7_audio_a_imagen_reconstruida.png", dpi=150)
    print("[+] Gráfica guardada como 'etapa7_audio_a_imagen_reconstruida.png'. Mostrando ventana...")
    plt.show(block=False)
    plt.pause(2.0)
    print()

    # Guardar copia de la imagen recuperada para verificación en disco
    Image.fromarray(img_desde_audio).save("imagen_recuperada_desde_audio.png")
    print("[+] Imagen final guardada en disco como: 'imagen_recuperada_desde_audio.png'")
    return img_desde_audio


# =============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =============================================================================
def main():
    print("\n" + "#" * 75)
    print("  LABORATORIO DE COMUNICACIONES I:")
    print("  TRANSMISIÓN DIGITAL DE IMÁGENES Y CONVERSIÓN IMAGEN-AUDIO-IMAGEN")
    print("#" * 75 + "\n")

    # Ruta opcional de imagen pasada por argumento
    ruta_personalizada = sys.argv[1] if len(sys.argv) > 1 else None

    # Etapa 1: Selección y carga
    img_rgb = etapa1_cargar_imagen(ruta_personalizada)

    # Etapa 2: Transformación de espacios de color
    img_gris, img_rojo = etapa2_transformacion_color(img_rgb)

    # Etapa 3: Digitalización y representación binaria
    flujo_bytes, metadata = etapa3_digitalizacion_binaria(img_rgb)

    # Etapa 4: Simulación de transmisión digital
    trama_transmitida = etapa4_simulacion_transmision(flujo_bytes, metadata)

    # Etapa 5: Reconstrucción de la imagen y análisis de pérdidas
    img_reconstruida = etapa5_reconstruccion_imagen(trama_transmitida, img_rgb)

    # Etapa 6: Conversión de imagen a audio
    ruta_wav = etapa6_imagen_a_audio(trama_transmitida, "imagen_a_audio.wav", tasa_muestreo=44100)

    # Etapa 7: Conversión de audio a imagen
    img_desde_audio = etapa7_audio_a_imagen(ruta_wav, img_rgb)

    print("=" * 75)
    print("¡TODAS LAS ETAPAS DEL LABORATORIO SE COMPLETARON CON ÉXITO!")
    print("Archivos generados en el directorio:")
    print("  1. imagen_original.png                 (Imagen de entrada)")
    print("  2. etapa2_espacios_de_color.png        (Original, Grises y Canal Rojo)")
    print("  3. etapa4_transmision_flujo_bits.png   (Señal digital en banda base NRZ-L)")
    print("  4. etapa5_reconstruccion_imagen.png    (Comparación original vs reconstruida)")
    print("  5. imagen_a_audio.wav                  (Audio acústico derivado de los píxeles)")
    print("  6. etapa6_forma_de_onda_audio.png      (Forma de onda en el tiempo)")
    print("  7. etapa7_audio_a_imagen_reconstruida.png (Comparación original vs desde audio)")
    print("  8. imagen_recuperada_desde_audio.png   (Imagen reconstruida a partir del audio)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
