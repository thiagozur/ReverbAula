import os
from pathlib import Path
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, Menu
import soundfile as sf
from scipy.signal import fftconvolve, resample
import sounddevice as sd

ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('blue')

class ReverbAula(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Reverb en base IR del aula')
        self.geometry('500x420')

        self.audio_path = None
        self.ir_path = None
        self.audio_dry = None
        self.audio_ir = None
        self.audio_process = None
        self.fs = 44100
        self.fs_ir = 44100
        self.mix_actual = 0.4
        self.target_mix = 0.4
        self.suavizado = 0.8
        self.hilo_decay = False

        self.carpeta_ir = Path(__file__).parent / 'IR'
        self.presets = {}
        self.escanear_presets()

        self.inicializar_ui()

        if self.presets:
            self.cambiar_preset(list(self.presets.keys())[0])
        else:
            self.lbl_nombre_preset.configure(text = 'No se encontraron IRs')
            self.btn_prev_preset.configure(state = 'disabled')
            self.btn_next_preset.configure(state = 'disabled')

    def escanear_presets(self):
        if not self.carpeta_ir.exists():
            self.carpeta_ir.mkdir(parents = True, exist_ok = True)
            return
        
        archivos_wav = sorted(self.carpeta_ir.glob('*.[wW][aA][vV]'))

        for ruta in archivos_wav:
            nombre = ruta.stem.replace('_', ' ').title()
            self.presets[nombre] = ruta

    def inicializar_ui(self):
        self.frame_preset = ctk.CTkFrame(self)
        self.frame_preset.pack(pady = (15, 5), padx = 20, fill = 'x')
        self.frame_preset.grid_columnconfigure(1, weight=1)

        self.btn_prev_preset = ctk.CTkButton(self.frame_preset, text = '◀', width = 40, command = self.preset_anterior)
        self.btn_prev_preset.grid(row = 0, column = 0, padx = 10, pady = 10)

        self.lbl_nombre_preset = ctk.CTkLabel(self.frame_preset, text = 'Buscando IRs...', font = ctk.CTkFont(size = 15, weight = 'bold'), cursor = 'hand2')
        self.lbl_nombre_preset.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'ew')
        self.lbl_nombre_preset.bind("<Button-1>", self.mostrar_menu_desplegable)

        self.btn_next_preset = ctk.CTkButton(self.frame_preset, text = '▶', width = 40, command = self.preset_siguiente)
        self.btn_next_preset.grid(row = 0, column = 2, padx = 10, pady = 10)

        self.frame_archivo = ctk.CTkFrame(self)
        self.frame_archivo.pack(pady = 15, padx = 20, fill = 'x')

        self.btn_cargar_audio = ctk.CTkButton(self.frame_archivo, text = 'Cargar archivo de audio', command = self.cargar_audio)
        self.btn_cargar_audio.grid(row = 0, column = 0, padx = 10, pady = 10)
        self.lbl_estado_audio = ctk.CTkLabel(self.frame_archivo, text = 'No hay ningún archivo cargado')
        self.lbl_estado_audio.grid(row = 0, column = 1, padx = 10, pady = 10)

        self.frame_parametros = ctk.CTkFrame(self)
        self.frame_parametros.pack(pady = 15, padx = 20, fill = 'x')

        self.lbl_decay = ctk.CTkLabel(self.frame_parametros, text = 'Escalado de decay: 1.0')
        self.lbl_decay.pack(anchor = 'w', padx = 15, pady = (10, 0))
        self.slider_decay = ctk.CTkSlider(self.frame_parametros, from_ = 0.1, to = 5.0, number_of_steps = 49, command = self.actualizar_lbl_decay)
        self.slider_decay.set(1.0)
        self.slider_decay.pack(fill = 'x', padx = 15, pady = (0, 10))

        self.lbl_mix = ctk.CTkLabel(self.frame_parametros, text = 'Mix: 40%')
        self.lbl_mix.pack(anchor = 'w', padx = 15, pady = (10, 0))
        self.slider_mix = ctk.CTkSlider(self.frame_parametros, from_ = 0.0, to = 1.0, number_of_steps = 29, command = self.actualizar_lbl_mix)
        self.slider_mix.set(self.mix_actual)
        self.slider_mix.pack(fill = 'x', padx = 15, pady = (0, 10))

        self.frame_playback = ctk.CTkFrame(self, fg_color = 'transparent')
        self.frame_playback.pack(pady = 5)

        self.btn_preview = ctk.CTkButton(self.frame_playback, text = 'Preescucha', state = 'disabled', command = self.iniciar_stream)
        self.btn_preview.grid(row = 0, column = 0, padx = 5)

        self.btn_parar = ctk.CTkButton(self.frame_playback, text = 'Detener reproducción', state = 'disabled', command = self.detener_stream, width = 60, fg_color = 'crimson', hover_color = 'darkred')
        self.btn_parar.grid(row = 0, column = 1, padx = 5)
        
        self.btn_guardar =ctk.CTkButton(self.frame_playback, text = 'Guardar audio procesado', state = 'disabled', command = self.guardar_audio)
        self.btn_guardar.grid(row = 0, column = 2, padx = 5)

    def actualizar_lbl_decay(self, val):
        self.lbl_decay.configure(text = f'Escalado de decay: {val:.2f}')
        
        if self.hilo_decay:
            return
        
        self.hilo_decay = True
        hilo = threading.Thread(target = self.procesar_audio, daemon = True)
        hilo.start()
    
    def actualizar_lbl_mix(self, val):
        self.lbl_mix.configure(text = f'Mix: {int(val*100)}%')

        self.target_mix = val

    def disparar_procesamiento(self, event = None):
        if not hasattr(self, 'audio_ir') or self.audio_ir is None:
            return
        
        if not hasattr(self, 'audio_dry') or self.audio_dry is None:
            return
        
        threading.Thread(target = self.procesar_audio, daemon = True).start()

    def cargar_audio(self):
        self.detener_stream()
        raw_filepath = filedialog.askopenfilename(filetypes = [('Archivos de audio', '*.wav *.flac *.mp3')])
        if raw_filepath:
            self.audio_path = Path(raw_filepath)
            self.audio_dry, self.fs = sf.read(self.audio_path, always_2d = True)

            if self.audio_dry.shape[1] == 1:
                self.audio_dry = np.hstack([self.audio_dry, self.audio_dry])

            self.lbl_estado_audio.configure(text=f'{self.audio_path.name} ({self.fs} Hz)')

            self.disparar_procesamiento()

    def mostrar_menu_desplegable(self, event):
        if not self.presets:
            return

        menu_contextual = Menu(self, tearoff = 0)

        fuente = ctk.CTkFont(size = 12)
        familia_fuente = fuente.cget('family')

        menu_contextual.configure(
            bg = '#2b2b2b' if ctk.get_appearance_mode() == 'Dark' else '#dbdbdb',
            fg = 'white' if ctk.get_appearance_mode() == 'Dark' else 'black',
            activebackground = '#1f6aa5',
            activeforeground = 'white',
            font = (familia_fuente, 11)
        )

        for nombre in self.presets.keys():
            menu_contextual.add_command(label = nombre, command = lambda name = nombre: self.cambiar_preset(name))

        menu_contextual.tk_popup(event.x_root, event.y_root)
    
    def cambiar_preset(self, preset):
        if preset not in self.presets:
            return
        
        self.lbl_nombre_preset.configure(text = preset)

        ruta_ir = self.presets[preset]
        try:
            audio_ir_raw, self.fs_ir = sf.read(ruta_ir, always_2d = True)

            audio_ir_plano = np.squeeze(audio_ir_raw)
            if audio_ir_plano.ndim > 1:
                audio_ir_plano = audio_ir_plano[:, 0]
            
            ind_pico = np.argmax(np.abs(audio_ir_plano))
            audio_ir_recortado = audio_ir_raw[ind_pico:]
            audio_ir_plano_recortado = audio_ir_plano[ind_pico:]

            umbral = 0.01 * np.max(np.abs(audio_ir_plano_recortado))
            ind_signal = np.where(np.abs(audio_ir_plano_recortado) > umbral)[0]

            if len(ind_signal) > 0:
                ultimo_ind = ind_signal[-1]
                margen = int(0.2 * self.fs_ir)
                corte = min(ultimo_ind + margen, len(audio_ir_recortado))
                self.audio_ir = audio_ir_recortado[:corte]
            else:
                self.audio_ir = audio_ir_recortado

            self.disparar_procesamiento()
        except Exception as e:
            print(f'Error al cargar la IR: {e}')

    def preset_anterior(self):
        if not self.presets:
            return
        nombres = list(self.presets.keys())
        actual = self.lbl_nombre_preset.cget('text')

        ind_actual = nombres.index(actual) if actual in nombres else 0
        ind_nuevo = (ind_actual - 1) % len(nombres)

        self.cambiar_preset(nombres[ind_nuevo])

    def preset_siguiente(self):
        if not self.presets:
            return
        
        nombres = list(self.presets.keys())
        actual = self.lbl_nombre_preset.cget('text')

        ind_actual = nombres.index(actual) if actual in nombres else 0
        ind_nuevo = (ind_actual + 1) % len(nombres)

        self.cambiar_preset(nombres[ind_nuevo])

    def start_processing_thread(self):
        if self.audio_dry is None:
            self.lbl_estado_audio.configure(text = '¡Cargue un archivo primero!')
            return 
        
        if self.audio_ir is None:
            print('¡No hay ninguna IR seleccionada en la carpeta!')
            return
        
        self.btn_procesar.configure(state = 'disabled', text = 'Procesando...')
        threading.Thread(target = self.procesar_audio, daemon = True).start()

    def procesar_audio(self):
        dry = self.audio_dry

        ir = self.audio_ir if self.audio_ir is not None else self.generar_ir_error()
        factor_decay = self.slider_decay.get()

        duracion_real = len(ir) / self.fs_ir
        decay_seg = duracion_real * factor_decay

        t = np.arange(len(ir)) / self.fs_ir

        if factor_decay <= 1:
            if factor_decay == 1:
                ir_modificada = ir.copy()
            else:
                envolvente = np.exp(- (5.0 / decay_seg) * t).reshape(-1, 1)
                ir_modificada = ir * envolvente
        else:
            muestras_ataque = int(0.060 * self.fs_ir)
            
            ataque = ir[:muestras_ataque].copy()
            cola = ir[muestras_ataque:].copy()
            
            muestras_cola = len(cola)
            muestras_cola_nuevas = int(muestras_cola * factor_decay)
            
            ind = np.linspace(0, muestras_cola - 1, muestras_cola)
            ind_nuevos = np.linspace(0, muestras_cola - 1, muestras_cola_nuevas)
            
            n_canales_ir = ir.shape[1]
            cola_estirada = np.zeros((muestras_cola_nuevas, n_canales_ir))
            
            for canal in range(n_canales_ir):
                cola_estirada[:, canal] = np.interp(ind_nuevos, ind, cola[:, canal])
            
            muestras_fade = int(0.040 * self.fs_ir)
            if muestras_fade < muestras_cola_nuevas:
                rampa_subida = np.linspace(0.0, 1.0, muestras_fade).reshape(-1, 1)
                cola_estirada[:muestras_fade] *= rampa_subida
            
            ir_modificada = np.vstack([ataque, cola_estirada])

        canales_wet = []
        n_canales = min(dry.shape[1], ir_modificada.shape[1])

        for c in range(n_canales):
            canal_conv = fftconvolve(dry[:, c], ir_modificada[:, c], mode = 'full')
            canales_wet.append(canal_conv)

        audio_wet = np.stack(canales_wet, axis = -1)

        pad_len = len(audio_wet) - len(dry)
        dry_padded = np.pad(dry, ((0, pad_len), (0, 0)), mode = 'constant')[:len(audio_wet), :n_canales]

        rms_dry = np.sqrt(np.mean(dry_padded**2))
        rms_wet = np.sqrt(np.mean(audio_wet**2))
        
        if rms_wet > 0 and rms_dry > 0:
            audio_wet = audio_wet * (rms_dry / rms_wet) * 0.4

        self.audio_dry_padded = dry_padded
        self.audio_process = audio_wet

        self.after(0, self.procesado_terminado)
    
    def procesado_terminado(self):
        self.btn_preview.configure(state = 'normal')
        self.btn_parar.configure(state = 'normal')
        self.btn_guardar.configure(state = 'normal')

        self.hilo_decay = False

        if hasattr(self, 'reproduciendo') and self.reproduciendo:
            if self.frame_actual >= len(self.audio_process):
                self.frame_actual = len(self.audio_process) - 1

    def iniciar_stream(self):
        if not hasattr(self, 'audio_process') or self.audio_process is None:
            print('No hay audio procesado para reproducir')
            return
        
        self.detener_stream()

        if not hasattr(self, 'frame_actual') or self.frame_actual >= len(self.audio_process):
            self.frame_actual = 0

        self.reproduciendo = True

        canales = self.audio_process.shape[1] if self.audio_process.ndim > 1 else 1
        fs = getattr(self, 'fs', 48000)

        def callback(outdata, frames, time, status):
            if status:
                print(status)

            if self.frame_actual >= len(self.audio_process):
                outdata[:frames] = 0
                raise sd.CallbackStop()

            chunk_size = min(len(self.audio_process) - self.frame_actual, frames) 
        
            if chunk_size > 0 and self.reproduciendo:
                bloque_dry = self.audio_dry_padded[self.frame_actual : self.frame_actual + chunk_size]
                bloque_wet = self.audio_process[self.frame_actual : self.frame_actual + chunk_size]

                self.mix_actual += self.suavizado * (self.target_mix - self.mix_actual)

                ganancia_dry = np.cos(self.mix_actual * (np.pi / 2))
                ganancia_wet = np.sin(self.mix_actual * (np.pi / 2))

                bloque_final = (ganancia_dry * bloque_dry) + (ganancia_wet * bloque_wet)

                bloque_final = np.clip(bloque_final, -1.0, 1.0)

                if canales == 1 and bloque_final.ndim == 1:
                    outdata[:chunk_size, 0] = bloque_final
                else:
                    outdata[:chunk_size] = bloque_final

                if chunk_size < frames:
                    outdata[chunk_size:] = 0

                self.frame_actual += chunk_size
            else:
                outdata[:frames] = 0
                raise sd.CallbackStop()
            
        try:
            self.stream = sd.OutputStream(
                samplerate = fs,
                channels = canales,
                callback = callback,
                blocksize = 1024
            )
            self.stream.start()
        except Exception as e:
            print(e)

    def detener_stream(self):
        self.reproduciendo = False
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(e)
            finally:
                self.stream = None
            
            self.frame_actual = 0
    
    def guardar_audio(self):
        if self.audio_process is not None:
            raw_filepath = filedialog.asksaveasfilename(defaultextension = '.wav', filetypes = [("WAV Audio", "*.wav")])
            if raw_filepath:
                path_guardado = Path(raw_filepath)
                sf.write(path_guardado, self.audio_process, self.fs)

if __name__ == '__main__':
    app = ReverbAula()
    app.mainloop()