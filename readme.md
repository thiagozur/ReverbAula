# Reverb402
Este es un procesador de audio standalone desarrollado en Python para la simulación de la reverb del aula 402 de la sede Caseros II de la UNTREF. El programa convoluciona un archivo de audio *dry* (sin procesar) con una respuesta al impulso medida in situ para lograr el efecto. Soporta preescucha en tiempo real.

## Prestaciones

* **Preescucha en tiempo real de la señal procesada** (con la opción de hacerlo en bucle).
* **Exportación del archivo de audio procesado en formato WAV**.
* **Visualización de la respuesta al impulso elegida**.
* **Sistema de selección de IRs estilo banco de *presets***: dentro de la carpeta "IR" están las respuestas al impulso de las cuatro mediciones del aula simulada. Dentro de la carpeta "IR" está la carpeta "stereo", con archivos IR en estéreo en base a los pares de mediciones izquierda-derecha en cada posición de escucha generados automáticamente por la aplicación. Además se incluyeron dos versiones estéreo *prealigned*, que fueron creadas manualmente con otro criterio de alineación y ajuste de nivel (por lo que tienen una sonoridad distinta).
* **Soporte de IRs adicionales**: si el usuario lo requiere, es posible agregar otras respuestas al impulso. Si son mono, simplemente debe agregarlas dentro de la carpeta "IR". Si son estéreo, debe agregarlas dentro de "IR/stereo". Si tiene por separado los canales izquierdo y derecho, puede agregarlos directamente a la carpeta "IR" pero debe respetar el siguiente formato de nombre *"NOMBRE_izquierda.wav"* y *"NOMBRE_derecha.wav"* estrictamente. Se recomienda tener poco silencio en el archivo antes y después del impulso para reducir tiempos de procesado. Las versiones estéreo a partir de los dos canales se generarán automáticamente al siguiente inicio de la aplicación.

## Parámetros modificables

* **Decay (factor de escalado temporal)**: se implementó un parámetro para modificar la respuesta al impulso original de la sala (0.1x - 5x). Al setearse en 1x se utiliza la IR original. Por debajo de ese valor, la IR se recorta con una envolvente exponencial. Por encima, se aplica un escalado temporal mediante *time-stretching* logrado con interpolación lineal.
* **Pre-delay (factor de escalado temporal)**: control del tiempo de retardo inicial (0 ms - 150ms).
* **Controles de filtrado**: se implementaron filtros High-Pass (20 Hz - 500 Hz) y Low-Pass (1 kHz - 20 kHz) que actúan sobre la señal procesada. Ambos son filtros Butterworth de orden 2 (-12 dB/oct). Si se setean al mínimo (en el caso del High-Pass) o al máximo (en el caso del Low-Pass) de frecuencia quedan desactivados por completo.
* **Mix**: este parámetro controla la proporción entre señal procesada y sin procesar a la salida. Un mix al 0% implica una totalidad de señal *dry* a la salida, mientras que al 100% implica una señal completamente *wet*. La señal procesada pasa previamente por un proceso de normalización por proporción (en base a los valores eficaces) para igualar su sonoridad con la de la señal original antes de mezclarlas.

## Dependencias necesarias

La aplicación utiliza las librerías customtkinter, numpy, soundfile, sounddevice, matplotlib y seaborn. Para poder utilizarla deberán instalarse corriendo el siguiente comando por terminal:

```bash
pip install customtkinter numpy soundfile sounddevice matplotlib seaborn
```