# -*- coding: utf-8 -*-

from utils import *  # 导入共享工具和依赖

class DemucsWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def run(self, audio_path):
        if not DEMUCS_AVAILABLE:
            self.error.emit("Demucs 库未找到。请通过 'pip install demucs' 安装。")
            return
        try:
            self.progress.emit("🚀 (GPU) Demucs 4音轨分离...")
            output_dir = os.path.join(os.path.dirname(audio_path), "separated_demucs")
            model_name = "htdemucs_ft"
            args = ["-n", model_name, "-o", output_dir, "--filename", "{stem}.{ext}", audio_path]
            demucs.separate.main(args)
            stems_dir = os.path.join(output_dir, model_name)
            stem_paths = {k: os.path.join(stems_dir, f"{k}.wav") for k in ['vocals', 'drums', 'bass', 'other']}
            for stem, path in stem_paths.items():
                if not os.path.exists(path):
                    self.error.emit(f"❌ 分离失败-未找到 {path} 文件。")
                    return
            self.finished.emit(stem_paths)
        except Exception as e:
            self.error.emit(f"❌ Demucs 运行时错误: {e}")

class BatchedMidiConversionWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def _convert_stft_to_midi_track(self, S_db, sr, params, freqs, track_name=""):
        conversion_mode = params['conversion_mode']
        ticks_per_beat = 480
        tempo = 500000
        hop_length = int(sr * params['frame_ms'] / 1000)
        hop_ticks = int(mido.second2tick(hop_length / sr, ticks_per_beat, tempo)) or 1

        self.progress.emit(f"   - [{track_name}] 正在将音符写入MIDI轨道 (模式: {conversion_mode})...")
        track = mido.MidiTrack()
        track.name = track_name
        active_notes = {}
        time_since_last_event = 0

        for t_idx in range(S_db.shape[1]):
            frame = S_db[:, t_idx]
            peak_indices = np.where(frame > params['threshold_db'])[0]
            notes_this_frame = set()
            velocities_this_frame = {}
            if len(peak_indices) > 0:
                for p_idx in peak_indices:
                    freq = freqs[p_idx]
                    if freq > 0:
                        midi_note = int(round(librosa.hz_to_midi(freq)))
                        if 21 <= midi_note <= 108:
                            notes_this_frame.add(midi_note)
                            db_volume = frame[p_idx]
                            velocity = int(np.interp(db_volume, [params['threshold_db'], 0], [30, 127]))
                            velocity = max(30, min(127, velocity))
                            velocities_this_frame[midi_note] = max(velocities_this_frame.get(midi_note, 0), velocity)

            if conversion_mode == 'continuous':
                notes_to_turn_off = set(active_notes.keys()) - notes_this_frame
                if notes_to_turn_off:
                    for note in sorted(list(notes_to_turn_off)):
                        track.append(mido.Message('note_off', note=note, velocity=64, time=time_since_last_event))
                        time_since_last_event = 0
                        del active_notes[note]

                notes_to_turn_on = notes_this_frame - set(active_notes.keys())
                if notes_to_turn_on:
                    for note in sorted(list(notes_to_turn_on)):
                        velocity = velocities_this_frame.get(note, 64)
                        track.append(mido.Message('note_on', note=note, velocity=velocity, time=time_since_last_event))
                        time_since_last_event = 0
                        active_notes[note] = velocity
                time_since_last_event += hop_ticks
            else:  # Snapshot mode
                if notes_this_frame:
                    note_list = sorted(list(notes_this_frame))
                    for i, note in enumerate(note_list):
                        track.append(mido.Message('note_on', note=note, velocity=velocities_this_frame.get(note, 64),
                                                  time=0 if i > 0 else time_since_last_event))
                    time_since_last_event = 0
                    track.append(mido.Message('note_off', note=note_list[0], velocity=64, time=hop_ticks))
                    if len(note_list) > 1:
                        for note in note_list[1:]:
                            track.append(mido.Message('note_off', note=note, velocity=64, time=0))
                else:
                    time_since_last_event += hop_ticks

        if conversion_mode == 'continuous' and active_notes:
            note_list = sorted(list(active_notes.keys()))
            if note_list:
                track.append(mido.Message('note_off', note=note_list[0], velocity=64, time=time_since_last_event))
                if len(note_list) > 1:
                    for note in note_list[1:]:
                        track.append(mido.Message('note_off', note=note, velocity=64, time=0))
        return track

    def _create_and_save_spectrogram(self, S_db_batch, sr, hop_length, output_folder):
        try:
            self.progress.emit("📊 正在生成合成频谱图...")
            colors = ['Reds', 'Blues', 'Greens', 'Purples']
            stems = ['vocals', 'drums', 'bass', 'other']
            height, width = S_db_batch[0].shape
            dpi = 100
            fig_width, fig_height = width / dpi, height / dpi
            final_image = Image.new('RGBA', (width, height), (0, 0, 0, 255))

            for i in range(len(stems)):
                self.progress.emit(f"   - 正在绘制 {stems[i]} ({colors[i]}) 的频谱...")
                S_db_stem = S_db_batch[i]
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

                #将Figure和Axes的背景都设为完全透明
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)

                librosa.display.specshow(S_db_stem, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log',
                                         cmap=plt.get_cmap(colors[i]), ax=ax)
                ax.axis('off')
                buf = io.BytesIO()
                fig.savefig(buf, format='png', transparent=True, dpi=dpi)
                buf.seek(0)
                stem_image = Image.open(buf)
                final_image = Image.alpha_composite(final_image, stem_image)
                buf.close()
                plt.close(fig)

            output_path = os.path.join(output_folder, "频谱图_combined.png")
            final_image.save(output_path)
            self.progress.emit(f"✅ 合成频谱图已保存: {os.path.basename(output_path)}")
        except Exception as e:
            self.progress.emit(f"❌ 生成频谱图时出错: {e}")

    def run(self, stem_paths, midi_params, gains_db, output_folder, generate_spectrogram):
        try:
            os.makedirs(output_folder, exist_ok=True)
            self.progress.emit(f"📂 输出目录已创建: {output_folder}")
            stems = ['vocals', 'drums', 'bass', 'other']
            audio_batch, sr, max_len = [], None, 0

            self.progress.emit("🔄 正在加载音轨并应用增益...")
            for stem in stems:
                path = stem_paths.get(stem)
                if path and os.path.exists(path):
                    y, current_sr = librosa.load(path, sr=None)
                    if sr is None:
                        sr = current_sr
                    if sr != current_sr:
                        self.error.emit(f"❌ 错误: 音轨采样率不一致({sr} vs {current_sr})。")
                        return

                    # 应用增益
                    gain_db = gains_db.get(stem, 0)
                    gain_linear = 10.0 ** (gain_db / 20.0)
                    y = y * gain_linear
                    self.progress.emit(f"   - 已应用 {gain_db} dB 增益到 {stem} 音轨")

                    audio_batch.append(y)
                    if len(y) > max_len:
                        max_len = len(y)
                else:
                    audio_batch.append(np.array([]))
                    self.progress.emit(f"⚠️ 警告: 未找到 {stem} 音轨文件，将生成空MIDI。")

            padded_batch = [np.pad(y, (0, max_len - len(y))) if len(y) > 0 else np.zeros(max_len) for y in audio_batch]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.progress.emit(f"🚀 使用 {device.upper()} 对所有音轨进行批量频谱计算 (STFT)...")
            hop_length = int(sr * midi_params['frame_ms'] / 1000)
            n_fft = hop_length * 2
            batch_tensor = torch.from_numpy(np.stack(padded_batch)).to(device)
            window = torch.hann_window(n_fft).to(device)
            stft_result_batch = torch.stft(batch_tensor, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                                           window=window, center=True, return_complex=True)
            S_abs_gpu_batch = stft_result_batch.abs()
            S_abs_cpu_np_batch = S_abs_gpu_batch.cpu().numpy()
            S_db_batch = librosa.amplitude_to_db(S_abs_cpu_np_batch, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            self.progress.emit("✅ 批量计算完成。")

            if generate_spectrogram:
                self._create_and_save_spectrogram(S_db_batch, sr, hop_length, output_folder)

            combined_midi = mido.MidiFile(type=1)
            for i, stem_name in enumerate(stems):
                S_db_stem = S_db_batch[i]
                midi_track = self._convert_stft_to_midi_track(S_db_stem, sr, midi_params, freqs, track_name=stem_name)
                individual_midi = mido.MidiFile(type=0)
                individual_midi.tracks.append(midi_track)
                individual_output_path = os.path.join(output_folder, f"{stem_name}.mid")
                individual_midi.save(individual_output_path)
                self.progress.emit(f"   - ✅ 单独MIDI文件已保存: {stem_name}.mid")
                combined_midi.tracks.append(midi_track)

            combined_output_path = os.path.join(output_folder, "combined.mid")
            combined_midi.save(combined_output_path)
            self.progress.emit(f"🎶✅ 合并的多音轨MIDI文件已保存: combined.mid")
            self.finished.emit(f"🎉 全部转换完成！\n文件已保存到文件夹:\n{output_folder}")
        except Exception as e:
            self.error.emit(f"❌ MIDI批量转换过程中发生错误: {e}")

class SingleMidiGenerationWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)

    def run(self, audio_data, sr, params):
        try:
            y, output_path, conversion_mode = audio_data, params['output_path'], params['conversion_mode']
            hop_length = int(sr * params['frame_ms'] / 1000)
            n_fft = hop_length * 2
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.progress.emit(f"🔬 使用 {device.upper()} 计算频谱 (FFT)...")
            y_tensor = torch.from_numpy(y).to(device)
            window = torch.hann_window(n_fft).to(device)
            stft_result = torch.stft(y_tensor, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                                     window=window, center=True, return_complex=True)
            S_abs_gpu = stft_result.abs()
            S_abs_cpu_np = S_abs_gpu.cpu().numpy()
            S_db = librosa.amplitude_to_db(S_abs_cpu_np, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            mid = mido.MidiFile(type=1)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            hop_ticks = int(mido.second2tick(hop_length / sr, mid.ticks_per_beat, 500000)) or 1
            self.progress.emit(f"✍️ 正在将音符写入MIDI (模式: {conversion_mode})...")
            active_notes, time_since_last_event = {}, 0

            for t_idx in range(S_db.shape[1]):
                frame = S_db[:, t_idx]
                peak_indices = np.where(frame > params['threshold_db'])[0]
                notes_this_frame, velocities_this_frame = set(), {}
                if len(peak_indices) > 0:
                    for p_idx in peak_indices:
                        freq = freqs[p_idx]
                        if freq > 0:
                            midi_note = int(round(librosa.hz_to_midi(freq)))
                            if 21 <= midi_note <= 108:
                                notes_this_frame.add(midi_note)
                                db_volume = frame[p_idx]
                                velocity = int(np.interp(db_volume, [params['threshold_db'], 0], [30, 127]))
                                velocity = max(30, min(127, velocity))
                                velocities_this_frame[midi_note] = max(velocities_this_frame.get(midi_note, 0),
                                                                       velocity)
                if conversion_mode == 'continuous':
                    notes_to_turn_off = set(active_notes.keys()) - notes_this_frame
                    if notes_to_turn_off:
                        for note in sorted(list(notes_to_turn_off)):
                            track.append(mido.Message('note_off', note=note, velocity=64, time=time_since_last_event))
                            time_since_last_event = 0
                            del active_notes[note]
                    notes_to_turn_on = notes_this_frame - set(active_notes.keys())
                    if notes_to_turn_on:
                        for note in sorted(list(notes_to_turn_on)):
                            velocity = velocities_this_frame.get(note, 64)
                            track.append(mido.Message('note_on', note=note, velocity=velocity, time=time_since_last_event))
                            time_since_last_event = 0
                            active_notes[note] = velocity
                    time_since_last_event += hop_ticks
                else:
                    if notes_this_frame:
                        note_list = sorted(list(notes_this_frame))
                        for i, note in enumerate(note_list):
                            track.append(mido.Message('note_on', note=note, velocity=velocities_this_frame.get(note, 64),
                                                      time=0 if i > 0 else time_since_last_event))
                        time_since_last_event = 0
                        track.append(mido.Message('note_off', note=note_list[0], velocity=64, time=hop_ticks))
                        if len(note_list) > 1:
                            for note in note_list[1:]:
                                track.append(mido.Message('note_off', note=note, velocity=64, time=0))
                    else:
                        time_since_last_event += hop_ticks
            if conversion_mode == 'continuous' and active_notes:
                note_list = sorted(list(active_notes.keys()))
                if note_list:
                    track.append(mido.Message('note_off', note=note_list[0], velocity=64, time=time_since_last_event))
                    if len(note_list) > 1:
                        for note in note_list[1:]:
                            track.append(mido.Message('note_off', note=note, velocity=64, time=0))
            mid.save(output_path)
            self.finished.emit(f"🎶 MIDI转换完成！文件已保存到:\n{output_path}")
        except Exception as e:
            self.finished.emit(f"❌ MIDI转换错误: {e}")

class WavRenderingWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)

    def run(self, params):
        try:
            tasks = params['tasks']
            attack_ms, release_ms, polyphony = params['attack_ms'], params['release_ms'], params['polyphony']
            sample_rate = 44100
            max_duration = 600.0  # 最大10分钟，防止过度分配

            if not tasks:
                self.finished.emit("🤷‍ 没有文件需要处理。")
                return

            total_files = len(tasks)
            self.progress.emit(f"🚀 开始批量渲染 {total_files} 个MIDI文件...")

            for i, task in enumerate(tasks):
                midi_path, output_path = task['midi_path'], task['output_path']
                self.progress.emit(f"--- [{i + 1}/{total_files}] 正在处理: {os.path.basename(midi_path)} ---")
                mid = mido.MidiFile(midi_path)
                notes = []
                open_notes = {}
                current_time_seconds = 0.0
                default_tempo = 500000
                tempo = next((msg.tempo for msg in mido.merge_tracks(mid.tracks) if msg.is_meta and msg.type == 'set_tempo'), default_tempo)
                
                for msg in mido.merge_tracks(mid.tracks):
                    current_time_seconds += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                    if msg.is_meta and msg.type == 'set_tempo':
                        tempo = msg.tempo
                    elif msg.type == 'note_on' and msg.velocity > 0:
                        open_notes[msg.note] = (current_time_seconds, msg.velocity)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in open_notes:
                            start_time, velocity = open_notes.pop(msg.note)
                            duration = current_time_seconds - start_time
                            if duration > 0.001:
                                notes.append({'note': msg.note, 'start': start_time, 'duration': duration, 'velocity': velocity})
                
                if open_notes:
                    end_of_track_time = current_time_seconds
                    for note, (start_time, velocity) in open_notes.items():
                        duration = end_of_track_time - start_time
                        if duration > 0.001:
                            notes.append({'note': note, 'start': start_time, 'duration': duration, 'velocity': velocity})
                
                if not notes:
                    self.progress.emit("   - 🤷‍ 在此MIDI文件中没有找到任何音符，跳过。")
                    continue

                self.progress.emit(f"   - 🎶 解析完成，共找到 {len(notes)} 个音符。")
                total_duration = min(max((n['start'] + n['duration'] for n in notes), default=0) + (release_ms / 1000.0) + 1.0, max_duration)
                total_samples = int(total_duration * sample_rate)
                
                # 使用单个缓冲，减少内存使用
                audio_buffer = np.zeros(total_samples, dtype=np.float32)

                for note_info in notes:
                    freq = midi_to_hz(note_info['note'])
                    amplitude = note_info['velocity'] / 127.0
                    note_duration_samples = int(note_info['duration'] * sample_rate)
                    if note_duration_samples == 0:
                        continue
                    release_extension_samples = int(release_ms / 1000.0 * sample_rate)
                    render_duration_samples = note_duration_samples + release_extension_samples
                    t = np.linspace(0, render_duration_samples / sample_rate, render_duration_samples, False, dtype=np.float32)
                    sine_wave = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
                    envelope = np.ones(render_duration_samples, dtype=np.float32)
                    attack_samples = min(int(attack_ms / 1000.0 * sample_rate), note_duration_samples)
                    if attack_samples > 0:
                        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
                    envelope[note_duration_samples:] = np.linspace(1.0, 0.0, release_extension_samples, dtype=np.float32)
                    enveloped_wave = sine_wave * envelope
                    start_sample = int(note_info['start'] * sample_rate)
                    end_sample = start_sample + len(enveloped_wave)
                    if end_sample <= total_samples:
                        audio_buffer[start_sample:end_sample] += enveloped_wave
                    # 无需voice_free_times，因为我们不管理复音缓冲

                if np.any(audio_buffer):
                    max_amp = np.max(np.abs(audio_buffer))
                    if max_amp > 1.0:
                        audio_buffer /= max_amp
                
                audio_int16 = (audio_buffer * 32767).astype(np.int16)
                write_wav(output_path, sample_rate, audio_int16)
                self.progress.emit(f"   - ✅ WAV渲染成功！文件已保存到: {os.path.basename(output_path)}")
            
            self.finished.emit(f"🎉 全部渲染完成！成功处理了 {total_files} 个文件。")
        except MemoryError as mem_err:
            self.finished.emit(f"❌ 内存不足: {mem_err}。建议降低Polyphony或处理较短MIDI文件。")
        except Exception as e:
            self.finished.emit(f"❌ 渲染过程中发生错误: {e}")