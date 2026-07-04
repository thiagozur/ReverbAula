import os
from pathlib import Path
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, Menu
from modules.ctk_knob import CTkKnob
import modules.motor_audio as dsp
import soundfile as sf
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('blue')

class ReverbAula(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Reverb en base IR del aula')

        ancho = 1000
        alto = 850
        ancho_pantalla = self.winfo_screenwidth()
        alto_pantalla = self.winfo_screenheight()

        x = int((ancho_pantalla / 2) - (ancho / 2))
        y = int((alto_pantalla / 2) - (alto / 2))

        self.geometry(f'{ancho}x{alto}+{x}+{y}')

        self.audio_path = None
        self.ir_path = None
        self.audio_dry = None
        self.audio_ir = None
        self.audio_process = None
        self.fs = 48000
        self.fs_ir = 48000
        self.mix_actual = 0.4
        self.target_mix = 0.4
        self.suavizado = 0.8
        self.hilo_decay = False
        self.playback_loop = False
        self._ultimo_tiempo_grafico = 0.0
        self.ir_modificada = None

        self.carpeta_ir = Path(__file__).parent / 'IR'
        self.carpeta_ir_stereo = self.carpeta_ir / 'stereo'
        self.presets = {}
        self.escanear_presets()

        self.inicializar_ui()

        if self.presets:
            self.cambiar_preset(list(self.presets.keys())[0])
        else:
            self.lbl_nombre_preset.configure(text = 'No se encontraron IRs')
            self.btn_prev_preset.configure(state = 'disabled')
            self.btn_next_preset.configure(state = 'disabled')

        self.protocol("WM_DELETE_WINDOW", self.onclose)

    def onclose(self):
        for alarm in self.tk.eval('after info').split():
            try:
                self.after_cancel(alarm)
            except Exception:
                pass

            try:
                plt.close('all')
            except Exception:
                pass
        
        self.destroy()

    def escanear_presets(self):
        if not self.carpeta_ir.exists():
            self.carpeta_ir.mkdir(parents = True, exist_ok = True)
            return
        
        if not self.carpeta_ir_stereo.exists():
            self.carpeta_ir_stereo.mkdir(parents = True, exist_ok = True)
            return

        archivos_wav = sorted(self.carpeta_ir.glob('*.[wW][aA][vV]'))

        wide_tuples = []
        for ruta_L in archivos_wav:
            if ruta_L.stem.endswith('_izquierda'):
                nombre_R = ruta_L.stem[:-10] + '_derecha' + ruta_L.suffix
                ruta_R = ruta_L.parent / nombre_R

                if ruta_R in archivos_wav:
                    wide_tuples.append((ruta_L, ruta_R))
        dsp.preparar_ir(self, make_wides = wide_tuples)

        archivos_wide = sorted(self.carpeta_ir_stereo.glob('*.[wW][aA][vV]'))

        count = 1
        for ruta in archivos_wav:
            nombre = str(count) + ' - ' + ruta.stem.replace('_', ' ').title()
            self.presets[nombre] = ruta
            count +=1
        
        for ruta in archivos_wide:
            nombre = str(count) + ' - ' + ruta.stem.replace('_', ' ').title()
            self.presets[nombre] = ruta
            count +=1

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

        self.frame_visualizador = ctk.CTkFrame(self, width = 550, height = 350, fg_color = '#1a1a1a', corner_radius = 10)
        self.frame_visualizador.pack_propagate(False)
        self.frame_visualizador.pack(fill = 'both', expand = True, padx = 15, pady = 15)

        sns.set_theme(style = 'darkgrid', rc = {
            'axes.facecolor': '#1a1a1a',
            'figure.facecolor': '#1a1a1a',
            'grid.color': '#2d2d2d',
            'axes.edgecolor': '#2d2d2d',
            'text.color': '#ffffff',
            'xtick.color': '#888888',
            'ytick.color': '#888888'
        })
        self.fig, self.ax = plt.subplots(figsize = (5, 2.5), facecolor = '#1a1a1a')

        self.ax.tick_params(colors = '#888888', labelsize = 9)
        self.ax.grid(True, color = '#333333', linestyle = '--', linewidth = 0.5)
        self.ax.set_title('Respuesta al impulso (modificada)', color = '#ffffff', fontsize = 10, pad = 10)
        self.ax.set_xlabel('Tiempo (s)', color = '#888888', fontsize = 9)
        self.ax.set_ylabel('Amplitud', color = '#888888', fontsize = 9)

        for spine in self.ax.spines.values():
            spine.set_color('#333333')

        self.canvas_grafico = FigureCanvasTkAgg(self.fig, master = self.frame_visualizador)
        self.widget_grafico = self.canvas_grafico.get_tk_widget()
        self.widget_grafico.pack(fill = 'both', expand = True, padx = 10, pady = 10)

        self.fig.tight_layout()

        self.frame_parametros = ctk.CTkFrame(self)
        self.frame_parametros.pack(pady = 15, padx = 20, fill = 'x')

        self.frame_knobs = ctk.CTkFrame(self.frame_parametros, fg_color = 'transparent')
        self.frame_knobs.pack(fill = 'x', padx = 18, pady = 15)

        self.contenedor_decay = ctk.CTkFrame(self.frame_knobs, fg_color = 'transparent', width = 120)
        self.contenedor_decay.pack(side = 'left', fill = 'both', expand = True, padx = 10, pady = 10)
        self.lbl_decay = ctk.CTkLabel(self.contenedor_decay, text = 'Decay\n1.0x', font = ctk.CTkFont(size = 11, weight = 'bold'), anchor = 'center', width = 120)
        self.lbl_decay.pack(pady = (0, 2))
        self.knob_decay = CTkKnob(self.contenedor_decay, from_ = 0.1, to = 5.0, step = 0.1, size = 70, command = self.actualizar_decay)
        self.knob_decay.pack(pady = 5)
        self.knob_decay.set(1.0)

        self.contenedor_predelay = ctk.CTkFrame(self.frame_knobs, fg_color = 'transparent', width = 120)
        self.contenedor_predelay.pack(side = 'left', fill = 'both', expand = True, padx = 10, pady = 10)
        self.lbl_predelay = ctk.CTkLabel(self.contenedor_predelay, text = 'Pre-delay\n0 ms', font = ctk.CTkFont(size = 11, weight = 'bold'), anchor = 'center', width = 120)
        self.lbl_predelay.pack(pady = (0, 2))
        self.knob_predelay = CTkKnob(self.contenedor_predelay, from_ = 0, to = 150, step = 1, size = 70, command = self.actualizar_predelay)
        self.knob_predelay.pack(pady = 5)
        self.knob_predelay.set(0)

        self.contenedor_hpf = ctk.CTkFrame(self.frame_knobs, fg_color = 'transparent', width = 120)
        self.contenedor_hpf.pack(side = 'left', fill = 'both', expand = True, padx = 10, pady = 10)
        self.lbl_hpf = ctk.CTkLabel(self.contenedor_hpf, text = 'Filtro High-Pass\n20 Hz', font = ctk.CTkFont(size = 11, weight = 'bold'), anchor = 'center', width = 120)
        self.lbl_hpf.pack(pady = (0, 2))
        self.knob_hpf = CTkKnob(self.contenedor_hpf, from_ = 20, to = 500, step = 1, size = 70, command = self.actualizar_filtros)
        self.knob_hpf.pack(pady = 5)
        self.knob_hpf.set(20)

        self.contenedor_lpf = ctk.CTkFrame(self.frame_knobs, fg_color = 'transparent', width = 120)
        self.contenedor_lpf.pack(side = 'left', fill = 'both', expand = True, padx = 10, pady = 10)
        self.lbl_lpf = ctk.CTkLabel(self.contenedor_lpf, text = 'Filtro Low-Pass\n20 kHz', font = ctk.CTkFont(size = 11, weight = 'bold'), anchor = 'center', width = 120)
        self.lbl_lpf.pack(pady = (0, 2))
        self.knob_lpf = CTkKnob(self.contenedor_lpf, from_ = 1000, to = 20000, step = 1, size = 70, command = self.actualizar_filtros)
        self.knob_lpf.pack(pady = 5)
        self.knob_lpf.set(20000)

        self.contenedor_mix = ctk.CTkFrame(self.frame_knobs, fg_color = 'transparent', width = 120)
        self.contenedor_mix.pack(side = 'left', fill = 'both', expand = True, padx = 10, pady = 10)
        self.lbl_mix = ctk.CTkLabel(self.contenedor_mix, text = 'Mix\n40%', font = ctk.CTkFont(size = 11, weight = 'bold'), anchor = 'center', width = 120)
        self.lbl_mix.pack(pady = (0, 2))
        self.knob_mix = CTkKnob(self.contenedor_mix, from_ = 0.0, to = 1.0, step = 0.01, size = 70, command = self.actualizar_mix)
        self.knob_mix.pack(pady = 5)
        self.knob_mix.set(0.4)

        self.frame_playback = ctk.CTkFrame(self, fg_color = 'transparent', height = 60)
        self.frame_playback.pack(fill = 'x', side = 'bottom', padx = 15, pady = 10)
        self.frame_playback.pack_propagate(False)

        self.btn_preview = ctk.CTkButton(self.frame_playback, text = 'Preescucha', state = 'disabled', command = self.iniciar_stream)
        self.btn_preview.pack(side = 'left', fill = 'none', expand = True, padx = 20)

        self.btn_parar = ctk.CTkButton(self.frame_playback, text = 'Detener reproducción', state = 'disabled', command = self.detener_stream, width = 60, fg_color = 'crimson', hover_color = 'darkred')
        self.btn_parar.pack(side = 'left', fill = 'none', expand = True, padx = 20)
        
        self.btn_guardar =ctk.CTkButton(self.frame_playback, text = 'Guardar audio procesado', state = 'disabled', command = self.guardar_audio)
        self.btn_guardar.pack(side = 'left', fill = 'none', expand = True, padx = 20)

        self.switch_loop = ctk.CTkSwitch(self.frame_playback, text = 'Loop de preescucha', command = self.toggle_loop, progress_color = '#1f538d')
        self.switch_loop.pack(side = 'left', fill = 'none', expand = True, padx = 20)

    def actualizar_grafico(self, ir_modificada):
        tiempo_actual = time.time()
        if tiempo_actual - self._ultimo_tiempo_grafico < 0.06:
            return
        self._ultimo_tiempo_grafico = tiempo_actual

        self.ax.clear()    

        for spine in self.ax.spines.values():
            spine.set_color('#333333')

        if ir_modificada is not None and len(ir_modificada) > 0:
            if len(ir_modificada.shape) > 1:
                y = ir_modificada[:, 0]
            else:
                y = ir_modificada

            fs_ir = getattr(self, 'fs_ir', 48000)
            
            step = max(1, len(y) // 1500)
            downsample_y = y[::step]
            downsample_x = (np.arange(len(y)) / fs_ir)[::step]

            self.ax.plot(downsample_x, downsample_y, color = '#007acc', linewidth = 0.2, alpha = 0.9)
            self.ax.fill_between(downsample_x, downsample_y, color = '#007acc', alpha = 0.5)
            
            max_amp = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
            self.ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
            self.ax.set_xlim(0, downsample_x[-1])
        
        self.ax.set_title('Respuesta al impulso (modificada)', color = '#ffffff', fontsize = 10, pad = 10)
        self.ax.set_xlabel('Tiempo (s)', color = '#888888', fontsize = 9)
        self.ax.set_ylabel('Amplitud', color = '#888888', fontsize = 9)

        try:
            self.canvas_grafico.draw()
        except Exception:
            pass
    
    def toggle_loop(self):
        val = self.switch_loop.get()
        if val == 1:
            self.playback_loop = True
        else:
            self.playback_loop = False

    def actualizar_filtros(self, val):
        freq_hpf = int(float(self.knob_hpf.get()))
        freq_lpf = int(float(self.knob_lpf.get()))

        self.lbl_hpf.configure(text = f'Filtro High-Pass\n{freq_hpf} Hz')
        self.lbl_lpf.configure(text = f'Filtro Low-Pass\n{round(freq_lpf / 1000) if float(freq_lpf / 1000).is_integer() else round(freq_lpf / 1000, 1)} kHz')

        if self.hilo_decay:
            return

        self.hilo_decay = True
        hilo = threading.Thread(target = self.procesar_audio)
        hilo.daemon = True
        hilo.start()

    def actualizar_predelay(self, val):
        self.lbl_predelay.configure(text = f'Pre-delay\n{int(float(val))} ms')      

        if self.hilo_decay:
            return
        
        self.hilo_decay = True
        hilo = threading.Thread(target = self.procesar_audio)
        hilo.daemon = True
        hilo.start()

    def actualizar_decay(self, val):
        self.lbl_decay.configure(text = f'Decay\n{round(val, 1)}x')
        
        if self.hilo_decay:
            return
        
        self.hilo_decay = True
        hilo = threading.Thread(target = self.procesar_audio, daemon = True)
        hilo.start()
    
    def actualizar_mix(self, val):
        self.lbl_mix.configure(text = f'Mix\n{int(val*100)}%')

        self.target_mix = val

    def disparar_procesamiento(self, event = None):
        if not hasattr(self, 'audio_ir') or self.audio_ir is None:
            return
        
        if not hasattr(self, 'audio_dry') or self.audio_dry is None:
            return
        
        if self.hilo_decay:
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

        try:
            self.audio_ir, self.fs_ir = dsp.preparar_ir(self, preset = preset)
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
        if self.audio_dry is None:
            self.hilo_decay = False
            return
        
        try:
            dry_padded, audio_wet, ir_modificada = dsp.procesar_convolucion_completa(
                dry = self.audio_dry,
                ir = self.audio_ir,
                factor_decay = self.knob_decay.get(),
                fs_ir = self.fs_ir,
                fs_audio = self.fs,
                ms_predelay = self.knob_predelay.get(),
                freq_hpf = self.knob_hpf.get(),
                freq_lpf = self.knob_lpf.get()
            )

            self.ir_modificada = ir_modificada
            self.audio_dry_padded = dry_padded
            self.audio_process = audio_wet

            self.after(0, self.procesado_terminado)
        except Exception as e:
            print(e)
            self.hilo_decay = False
    
    def procesado_terminado(self):
        if self.btn_preview.cget('state') != 'normal':
            self.btn_preview.configure(state = 'normal')
        
        if self.btn_parar.cget('state') != 'normal':
            self.btn_parar.configure(state = 'normal')

        if self.btn_guardar.cget('state') != 'normal':
            self.btn_guardar.configure(state = 'normal')

        self.actualizar_grafico(self.ir_modificada)

        if hasattr(self, 'reproduciendo') and self.reproduciendo:
            if self.frame_actual >= len(self.audio_process):
                if self.playback_loop:
                    self.frame_actual = 0
                else:
                    self.frame_actual = len(self.audio_process) - 1

        self.hilo_decay = False

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
                if self.playback_loop and self.reproduciendo:
                    self.frame_actual = 0
                else:
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
                    if self.playback_loop and self.reproduciendo:
                        frames_faltantes = frames - chunk_size
                        self.frame_actual = 0

                        bloque_dry_nuevo = self.audio_dry_padded[:frames_faltantes]
                        bloque_wet_nuevo = self.audio_process[:frames_faltantes]

                        bloque_final_nuevo = (ganancia_dry * bloque_dry_nuevo) + (ganancia_wet * bloque_wet_nuevo)
                        bloque_final_nuevo = np.clip(bloque_final_nuevo, -1.0, 1.0)

                        if canales == 1 and bloque_final_nuevo.ndim == 1:
                            outdata[chunk_size:, 0] = bloque_final_nuevo
                        else:
                            outdata[chunk_size:] = bloque_final_nuevo

                        self.frame_actual += frames_faltantes
                    else:
                        outdata[chunk_size:] = 0
                        self.frame_actual += chunk_size
                else:
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