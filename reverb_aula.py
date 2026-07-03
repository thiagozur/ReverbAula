import os
from pathlib import Path
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
import soundfile as sf
from scipy.signal import fftconvolve
import sounddevice as sd

ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('blue')

class ReverbAula(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Reverb en base IR del aula')
        self.geometry('500x400')

        self.audio_path = None
        self.ir_path = None
        self.audio_dry = None
        self.audio_ir = None
        self.audio_process = None
        self.fs = 44100
        self.fs_ir = 44100

        self.inicializar_ui()

    def inicializar_ui(self):
        self.frame_archivo = ctk.CTkFrame(self)
        self.frame_archivo.pack(pady = 15, padx = 20, fill = 'x')

        self.btn_cargar_audio = ctk.CTkButton(self.frame_archivo, text = 'Cargar archivo de audio', command = self.cargar_audio)
        self.btn_cargar_audio.grid(row = 0, column = 0, padx = 10, pady = 10)
        self.lbl_estado_audio = ctk.CTkLabel(self.frame_archivo, text = 'No hay ningún archivo cargado')
        self.lbl_estado_audio.grid(row = 0, column = 1, padx = 10, pady = 10)

        self.btn_cargar_ir = ctk.CTkButton(self.frame_archivo, text = 'Cargar IR', command = self.cargar_ir)
        self.btn_cargar_ir.grid(row = 1, column = 0, padx = 10, pady = 10)
        self.lbl_estado_ir = ctk.CTkLabel(self.frame_archivo, text = 'No hay ninguna IR cargada (se usa una artificial)')
        self.lbl_estado_ir.grid(row = 1, column = 1, padx = 10, pady = 10)

        self.frame_parametros = ctk.CTkFrame(self)
        self.frame_parametros.pack(pady = 15, padx = 20, fill = 'x')

        self.lbl_decay = ctk.CTkLabel(self.frame_parametros, text = 'Escalado de decay: 1.0')
        self.lbl_decay.pack(anchor = 'w', padx = 15, pady = (10, 0))
        self.slider_decay = ctk.CTkSlider(self.frame_parametros, from_ = 0.1, to = 3.0, number_of_steps = 29, command = self.actualizar_lbl_decay)
        self.slider_decay.set(1.0)
        self.slider_decay.pack(fill = 'x', padx = 15, pady = (0, 10))

        self.lbl_mix = ctk.CTkLabel(self.frame_parametros, text = 'Mix: 40%')
        self.lbl_mix.pack(anchor = 'w', padx = 15, pady = (10, 0))
        self.slider_mix = ctk.CTkSlider(self.frame_parametros, from_ = 0.0, to = 1.0, number_of_steps = 29, command = self.actualizar_lbl_mix)
        self.slider_mix.set(0.4)
        self.slider_mix.pack(fill = 'x', padx = 15, pady = (0, 10))

        self.btn_procesar = ctk.CTkButton(self, text = 'Aplicar procesamiento', command = self.start_processing_thread, fg_color = 'green', hover_color = 'darkgreen')
        self.btn_procesar.pack(pady = 10)

        self.frame_playback = ctk.CTkFrame(self, fg_color = 'transparent')
        self.frame_playback.pack(pady = 5)

        self.btn_preview = ctk.CTkButton(self.frame_playback, text = 'Preescucha', state = 'disabled', command = self.escuchar_preview)
        self.btn_preview.grid(row = 0, column = 0, padx = 5)

        self.btn_parar = ctk.CTkButton(self.frame_playback, text = 'Detener reproducción', state = 'disabled', command = self.stop_audio, width = 60, fg_color = 'crimson', hover_color = 'darkred')
        self.btn_parar.grid(row = 0, column = 1, padx = 5)
        
        self.btn_guardar =ctk.CTkButton(self.frame_playback, text = 'Guardar audio procesado', state = 'disabled', command = self.guardar_audio)
        self.btn_guardar.grid(row = 0, column = 2, padx = 5)

    def actualizar_lbl_decay(self, val):
        self.lbl_decay.configure(text = f'Escalado de decay: {val:.2f}')
    
    def actualizar_lbl_mix(self, val):
        self.lbl_mix.configure(text = f'Mix: {int(val*100)}%')

    def cargar_audio(self):
        raw_filepath = filedialog.askopenfilename(filetypes = [('Archivos de audio', '*.wav *.flac *.mp3')])
        if raw_filepath:
            self.audio_path = Path(raw_filepath)
            self.audio_dry, self.fs = sf.read(self.audio_path, always_2d = True)
            self.lbl_estado_audio.configure(text=f'{self.audio_path.name} ({self.fs} Hz)')

    def cargar_ir(self):
        raw_filepath = filedialog.askopenfilename(filetypes = [('Archivos de audio', '*.wav *.flac *.mp3')])
        if raw_filepath:
            self.ir_path = Path(raw_filepath)
            audio_ir_raw, self.fs_ir = sf.read(self.ir_path, always_2d = True)

            audio_ir_plano = np.squeeze(audio_ir_raw)
            if audio_ir_plano.ndim > 1:
                audio_ir_plano = audio_ir_plano[:, 0]
            
            pico_index = np.argmax(np.abs(audio_ir_plano))
            audio_ir_recortado = audio_ir_raw[pico_index:]
            audio_ir_plano_recortado = audio_ir_plano[pico_index:]

            umbral = 0.01 * np.max(np.abs(audio_ir_plano_recortado))
            ind_signal = np.where(np.abs(audio_ir_plano_recortado) > umbral)[0]

            if len(ind_signal) > 0:
                ultimo_ind = ind_signal[-1]
                margen = int(0.2 * self.fs_ir)
                corte = min(ultimo_ind + margen, len(audio_ir_recortado))
                self.audio_ir = audio_ir_recortado[:corte]
            else:
                self.audio_ir = audio_ir_recortado

            self.lbl_estado_ir.configure(text = f'{self.ir_path.name} ({self.fs_ir} Hz)')

    def generar_ir_error(self):
        t = np.linspace(0, 2.0, int(self.fs * 2.0), False)
        ruido = np.random.randn(len(t), 1)
        envolvente = np.exp(-3 * t).reshape(-1, 1)
        return ruido * envolvente
    
    def start_processing_thread(self):
        if self.audio_dry is None:
            self.lbl_estado_audio.configure(text = '¡Cargue un archivo primero!')
            return
        self.btn_procesar.configure(state = 'disabled', text = 'Procesando...')
        threading.Thread(target = self.procesar_audio, daemon = True).start()

    def procesar_audio(self):
        ir = self.audio_ir if self.audio_ir is not None else self.generar_ir_error()

        dry = self.audio_dry

        factor_decay = self.slider_decay.get()
        t = np.arange(len(ir)) / self.fs

        envolvente = np.exp(- (5.0 / factor_decay) * t).reshape(-1, 1)
        ir_modificada = ir * envolvente

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

        mix = self.slider_mix.get()

        ganancia_dry = np.cos(mix * (np.pi / 2))
        ganancia_wet = np.sin(mix * (np.pi / 2))

        self.audio_process = (ganancia_dry * dry_padded) + (ganancia_wet * audio_wet)

        val_max = np.max(np.abs(self.audio_process))
        if val_max > 0:
            self.audio_process /= val_max

        self.after(0, self.procesado_terminado)
    
    def procesado_terminado(self):
        self.btn_procesar.configure(state = 'normal', text = 'Aplicar procesamiento')
        self.btn_preview.configure(state = 'normal')
        self.btn_parar.configure(state = 'normal')
        self.btn_guardar.configure(state = 'normal')

    def escuchar_preview(self):
        if self.audio_process is not None:
            sd.stop()
            sd.play(self.audio_process, self.fs)

    def stop_audio(self):
        sd.stop()
    
    def guardar_audio(self):
        if self.audio_process is not None:
            raw_filepath = filedialog.asksaveasfilename(defaultextension = '.wav', filetypes = [("WAV Audio", "*.wav")])
            if raw_filepath:
                path_guardado = Path(raw_filepath)
                sf.write(path_guardado, self.audio_process, self.fs)

if __name__ == '__main__':
    app = ReverbAula()
    app.mainloop()