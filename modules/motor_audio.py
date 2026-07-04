import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
import soundfile as sf
from pathlib import Path

def modificar_decay_ir(ir, factor_decay, fs_ir):
    duracion_real = len(ir) / fs_ir
    decay_seg = duracion_real * factor_decay
    t = np.arange(len(ir)) / fs_ir

    if factor_decay <= 1:
        if factor_decay == 1:
            return ir.copy()
        envolvente = np.exp(- (5.0 / decay_seg) * t).reshape(-1, 1)
        return ir * envolvente
    else:
        muestras_ataque = int(0.060 * fs_ir)
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
        
        muestras_fade = int(0.040 * fs_ir)
        if muestras_fade < muestras_cola_nuevas:
            rampa_subida = np.linspace(0.0, 1.0, muestras_fade).reshape(-1, 1)
            cola_estirada[:muestras_fade] *= rampa_subida
        
        return np.vstack([ataque, cola_estirada])

def procesar_convolucion_completa(dry, ir, factor_decay, fs_ir, fs_audio, ms_predelay, freq_hpf, freq_lpf):
    ir_modificada = modificar_decay_ir(ir, factor_decay, fs_ir)
    
    canales_wet = []
    n_canales = min(dry.shape[1], ir_modificada.shape[1])
    for c in range(n_canales):
        canal_conv = fftconvolve(dry[:, c], ir_modificada[:, c], mode='full')
        canales_wet.append(canal_conv)
    audio_wet = np.stack(canales_wet, axis=-1)

    muestras_delay = int((ms_predelay * fs_audio) / 1000)
    if muestras_delay > 0:
        silencio = np.zeros((muestras_delay, n_canales))
        audio_wet = np.vstack([silencio, audio_wet])

    nyquist = fs_audio / 2.0
    if freq_hpf > 20:
        sos_hp = butter(2, freq_hpf / nyquist, btype='highpass', output='sos')
        for c in range(n_canales):
            audio_wet[:, c] = sosfilt(sos_hp, audio_wet[:, c])

    if freq_lpf < 20000:
        sos_lp = butter(2, freq_lpf / nyquist, btype='lowpass', output='sos')
        for c in range(n_canales):
            audio_wet[:, c] = sosfilt(sos_lp, audio_wet[:, c])

    pad_len = len(audio_wet) - len(dry)
    dry_padded = np.pad(dry, ((0, pad_len), (0, 0)), mode='constant')[:len(audio_wet), :n_canales]

    rms_dry = np.sqrt(np.mean(dry_padded**2))
    rms_wet = np.sqrt(np.mean(audio_wet**2))
    
    if rms_wet > 0 and rms_dry > 0:
        audio_wet = audio_wet * (rms_dry / rms_wet) * 0.4

    return dry_padded, audio_wet, ir_modificada

def preparar_ir(self, preset = None, make_wides = None):
        if make_wides is not None and len(make_wides) > 0:
            for par in make_wides:
                ruta_ir_L, ruta_ir_R = par[0], par[1]

                ruta_base = Path(ruta_ir_L)
                destino = ruta_base.parent / 'stereo'
                nombre = f'{ruta_base.stem.split("_")[0]}_wide.wav'
                ruta_salida = destino / nombre

                if ruta_salida.exists():
                    continue

                try:
                    ir_L_raw, fs_ir_L = sf.read(ruta_ir_L, always_2d = True)
                    ir_L_plano = np.squeeze(ir_L_raw)

                    ir_R_raw, fs_ir_R = sf.read(ruta_ir_R, always_2d = True)
                    ir_R_plano = np.squeeze(ir_R_raw)

                    if fs_ir_L != fs_ir_R:
                        raise Exception('Las frecuencias de muestreo de los canales no son iguales')
                    else:
                        fs_ir_stereo = fs_ir_L
                    
                    if ir_L_plano.ndim > 1:
                        ir_L_plano = ir_L_plano[:, 0]

                    if ir_R_plano.ndim > 1:
                        ir_R_plano = ir_R_plano[:, 0]

                    ind_pico_L = np.argmax(np.abs(ir_L_plano))
                    ir_L_recortado = ir_L_raw[ind_pico_L:]
                    ir_L_plano_recortado = ir_L_plano[ind_pico_L:]

                    ind_pico_R = np.argmax(np.abs(ir_R_plano))
                    ir_R_recortado = ir_R_raw[ind_pico_R:]
                    ir_R_plano_recortado = ir_R_plano[ind_pico_R:]

                    umbral_L = 0.01 * np.max(np.abs(ir_L_plano_recortado))
                    ind_signal_L = np.where(np.abs(ir_L_plano_recortado) > umbral_L)[0]

                    umbral_R = 0.01 * np.max(np.abs(ir_R_plano_recortado))
                    ind_signal_R = np.where(np.abs(ir_R_plano_recortado) > umbral_R)[0]

                    if len(ind_signal_L) > 0:
                        ultimo_ind_L = ind_signal_L[-1]
                        margen_L = int(0.2 * fs_ir_stereo)
                        corte_L = min(ultimo_ind_L + margen_L, len(ir_L_recortado))
                        ir_L = ir_L_recortado[:corte_L]
                    else:
                        ir_L = ir_L_recortado

                    if len(ind_signal_R) > 0:
                        ultimo_ind_R = ind_signal_R[-1]
                        margen_R = int(0.2 * fs_ir_stereo)
                        corte_R = min(ultimo_ind_R + margen_R, len(ir_R_recortado))
                        ir_R = ir_R_recortado[:corte_R]
                    else:
                        ir_R = ir_R_recortado

                    ir_L_final = ir_L.flatten().astype(np.float32)
                    ir_R_final = ir_R.flatten().astype(np.float32)

                    len_ir_L = len(ir_L_final)
                    len_ir_R = len(ir_R_final)
                    max_len = max(len_ir_L, len_ir_R)

                    if len_ir_L < max_len:
                        ir_L_final = np.pad(ir_L_final, (0, max_len - len_ir_L), mode = 'constant')
                    if len_ir_R < max_len:
                        ir_R_final = np.pad(ir_R_final, (0, max_len - len_ir_R), mode = 'constant')

                    ir = np.column_stack((ir_L_final, ir_R_final))

                    sf.write(ruta_salida, ir, fs_ir_stereo, format = 'WAV', subtype = 'PCM_24')

                except Exception as e:
                    print(e)
                
            return
                
        ruta_ir = self.presets[preset]

        try:
            audio_ir_raw, fs_ir = sf.read(ruta_ir, always_2d = True)

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
                margen = int(0.2 * fs_ir)
                corte = min(ultimo_ind + margen, len(audio_ir_recortado))
                audio_ir = audio_ir_recortado[:corte]
            else:
                audio_ir = audio_ir_recortado

            return audio_ir, fs_ir
        except Exception as e:
            print(e)