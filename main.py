# -*- coding: utf-8 -*-

"""
钢琴唱歌工具套件 (GUI版) v7.8 (内存优化版)
GPU加速 (Demucs+FFT), 双模式处理, 智能缓存, C盘保护
修复了合成频谱图只有drum音轨的bug。通过将Matplotlib的Figure和Axes背景均设为透明，实现了正确的Alpha混合效果。
“音轨增益”功能被正确实现。现在，增益滑块会直接调整对应音轨的音频信号强度，从而影响其在MIDI转换中的音符识别敏感度。
MIDI->WAV支持文件夹批量处理。
新增: 内存优化，修复长MIDI文件渲染时的内存分配错误。
依赖demucs, PyTorch (CUDA版), PyQt5, librosa, numpy, mido, scipy, matplotlib, Pillow
安装:
1. (NVIDIA GPU 用户) 访问 pytorch.org 获取并运行CUDA版的安装命令
2. pip install demucs PyQt5 librosa numpy mido scipy matplotlib Pillow
"""

from utils import *  # 导入共享工具和依赖
from workers import *  # 导入Worker类

setup_cache()  # 设置缓存

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('钢琴唱歌工具套件 v7.8 (内存优化版)')
        self.setGeometry(100, 100, 700, 800)
        main_layout = QVBoxLayout()
        tabs = QTabWidget()
        self.mp3_to_midi_tab = QWidget()
        self.midi_to_wav_tab = QWidget()
        tabs.addTab(self.mp3_to_midi_tab, "音频 转 MIDI")
        tabs.addTab(self.midi_to_wav_tab, "MIDI 转 WAV")

        self.init_mp3_to_midi_ui()
        self.init_midi_to_wav_ui()

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)

        main_layout.addWidget(tabs)
        main_layout.addWidget(log_group)
        self.setLayout(main_layout)

        if not DEMUCS_AVAILABLE:
            QMessageBox.warning(self, "依赖缺失", "未找到 'demucs' 库，音轨分离模式不可用。")

    def init_mp3_to_midi_ui(self):
        layout = QVBoxLayout()

        # 1. 文件选择
        file_group = QGroupBox("1. 输入音频文件")
        file_layout = QHBoxLayout()
        self.mp3_path_edit = QLineEdit()
        self.mp3_browse_btn = QPushButton("浏览...")
        self.mp3_browse_btn.clicked.connect(self.browse_mp3_file)
        file_layout.addWidget(self.mp3_path_edit)
        file_layout.addWidget(self.mp3_browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 2. 模式选择
        mode_selection_group = QGroupBox("2. 选择处理模式")
        mode_selection_layout = QHBoxLayout()
        self.mode_stem_radio = QRadioButton("音轨分离模式 (GPU加速)")
        self.mode_eq_radio = QRadioButton("整体EQ模式 (CPU)")
        self.mode_stem_radio.setChecked(True)
        self.mode_stem_radio.toggled.connect(self._update_ui_mode)
        if not DEMUCS_AVAILABLE:
            self.mode_stem_radio.setEnabled(False)
            self.mode_eq_radio.setChecked(True)
        self.force_rerun_checkbox = QCheckBox("强制重新分离")
        mode_selection_layout.addWidget(self.mode_stem_radio)
        mode_selection_layout.addWidget(self.mode_eq_radio)
        mode_selection_layout.addWidget(self.force_rerun_checkbox)
        mode_selection_group.setLayout(mode_selection_layout)
        layout.addWidget(mode_selection_group)

        # 3. 音轨增益参数 (分离模式)
        self.gain_group = QGroupBox("3. 音轨增益参数 (影响MIDI转换)")
        self.gain_group.setToolTip("调整各音轨的音量，以影响其在MIDI转换中的敏感度。\n高增益=更多音符，低增益=更少音符。")
        gain_layout = QFormLayout()
        self.stem_controls = {}
        stems = {'vocals': "人声", 'drums': "鼓", 'bass': "贝斯", 'other': "其他"}
        for key, name in stems.items():
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-40, 12)
            slider.setValue(0)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            spinbox = QSpinBox()
            spinbox.setRange(-40, 12)
            spinbox.setValue(0)
            spinbox.setSuffix(" dB")
            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)
            control_layout = QHBoxLayout()
            control_layout.addWidget(slider)
            control_layout.addWidget(spinbox)
            self.stem_controls[key] = {'slider': slider, 'spinbox': spinbox}
            gain_layout.addRow(f"{name}:", control_layout)
        self.gain_group.setLayout(gain_layout)
        layout.addWidget(self.gain_group)

        # 3. 频段增益 (EQ模式)
        self.eq_group = QGroupBox("3. 频段增益 (EQ) 参数")
        eq_layout = QVBoxLayout()
        self.eq_table = QTableWidget()
        self.eq_table.setColumnCount(3)
        self.eq_table.setHorizontalHeaderLabels(["起始(Hz)", "结束(Hz)", "增益(dB)"])
        self.eq_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.add_eq_band(0, 200, -10.0)
        self.add_eq_band(8000, 22000, -10.0)
        eq_btn_layout = QHBoxLayout()
        add_btn = QPushButton("添加")
        remove_btn = QPushButton("移除")
        add_btn.clicked.connect(lambda: self.add_eq_band(0, 0, 0.0))
        remove_btn.clicked.connect(self.remove_selected_eq_band)
        eq_btn_layout.addWidget(add_btn)
        eq_btn_layout.addWidget(remove_btn)
        eq_layout.addWidget(self.eq_table)
        eq_layout.addLayout(eq_btn_layout)
        self.eq_group.setLayout(eq_layout)
        layout.addWidget(self.eq_group)

        # 4. 通用高级参数
        adv_group = QGroupBox("4. 通用MIDI转换参数")
        adv_layout = QVBoxLayout()
        params_group = QGroupBox("基本参数")
        form_layout = QFormLayout()
        self.sensitivity_spin = QSpinBox()
        self.sensitivity_spin.setRange(-60, 0)
        self.sensitivity_spin.setValue(-30)
        self.sensitivity_spin.setSuffix(" dB")
        self.time_precision_spin = QSpinBox()
        self.time_precision_spin.setRange(10, 200)
        self.time_precision_spin.setValue(35)
        self.time_precision_spin.setSuffix(" ms")
        form_layout.addRow("灵敏度阈值:", self.sensitivity_spin)
        form_layout.addRow("时间精度:", self.time_precision_spin)
        params_group.setLayout(form_layout)
        adv_layout.addWidget(params_group)
        mode_group = QGroupBox("MIDI音符模式")
        mode_layout = QVBoxLayout()
        self.note_mode_continuous = QRadioButton("连续音符 (推荐)")
        self.note_mode_snapshot = QRadioButton("快照 (逐帧)")
        self.note_mode_continuous.setChecked(True)
        mode_layout.addWidget(self.note_mode_continuous)
        mode_layout.addWidget(self.note_mode_snapshot)
        mode_group.setLayout(mode_layout)
        adv_layout.addWidget(mode_group)
        self.generate_spec_checkbox = QCheckBox("生成频谱图 (可选)")
        self.generate_spec_checkbox.setToolTip("在音轨分离模式下，生成一张四色合成的频谱图。")
        adv_layout.addWidget(self.generate_spec_checkbox)
        adv_group.setLayout(adv_layout)
        layout.addWidget(adv_group)

        # 5. 开始按钮
        self.convert_btn = QPushButton("⭐ 开始转换")
        self.convert_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #337AB7; color: white;")
        self.convert_btn.clicked.connect(self.run_process_dispatcher)
        layout.addWidget(self.convert_btn)
        self.mp3_to_midi_tab.setLayout(layout)
        self._update_ui_mode()

    def _update_ui_mode(self):
        is_stem_mode = self.mode_stem_radio.isChecked()
        self.gain_group.setVisible(is_stem_mode)
        self.force_rerun_checkbox.setVisible(is_stem_mode)
        self.generate_spec_checkbox.setVisible(is_stem_mode)
        self.eq_group.setVisible(not is_stem_mode)

    def init_midi_to_wav_ui(self):
        layout = QVBoxLayout()
        file_group = QGroupBox("输入文件或文件夹")
        file_layout = QHBoxLayout()
        self.midi_path_edit = QLineEdit()
        self.midi_browse_btn = QPushButton("浏览文件")
        self.midi_browse_folder_btn = QPushButton("浏览文件夹")
        self.midi_browse_btn.clicked.connect(self.browse_midi_file)
        self.midi_browse_folder_btn.clicked.connect(self.browse_midi_folder)
        file_layout.addWidget(self.midi_path_edit)
        file_layout.addWidget(self.midi_browse_btn)
        file_layout.addWidget(self.midi_browse_folder_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        params_group = QGroupBox("渲染参数")
        form_layout = QFormLayout()
        self.attack_spin = QDoubleSpinBox()
        self.attack_spin.setRange(0.1, 200.0)
        self.attack_spin.setValue(5.0)
        self.attack_spin.setSuffix(" ms")
        self.release_spin = QDoubleSpinBox()
        self.release_spin.setRange(0.1, 500.0)
        self.release_spin.setValue(15.0)
        self.release_spin.setSuffix(" ms")
        self.polyphony_spin = QSpinBox()
        self.polyphony_spin.setRange(1, 256)
        self.polyphony_spin.setValue(64)
        form_layout.addRow("攻击时间(从音符出发到完全响) :", self.attack_spin)
        form_layout.addRow("释放时间(从完全响到不响):", self.release_spin)
        form_layout.addRow("最大复音数 (Polyphony):", self.polyphony_spin)
        params_group.setLayout(form_layout)
        layout.addWidget(params_group)

        self.render_to_wav_btn = QPushButton("开始渲染 MIDI -> WAV")
        self.render_to_wav_btn.setStyleSheet("font-size:16px;padding:10px;background-color:#008CBA;color:white;")
        self.render_to_wav_btn.clicked.connect(self.run_wav_rendering)
        layout.addWidget(self.render_to_wav_btn)
        layout.addStretch()
        self.midi_to_wav_tab.setLayout(layout)

    def add_eq_band(self, l, h, g):
        r = self.eq_table.rowCount()
        self.eq_table.insertRow(r)
        self.eq_table.setItem(r, 0, QTableWidgetItem(str(l)))
        self.eq_table.setItem(r, 1, QTableWidgetItem(str(h)))
        self.eq_table.setItem(r, 2, QTableWidgetItem(str(g)))

    def remove_selected_eq_band(self):
        r = self.eq_table.currentRow()
        if r >= 0:
            self.eq_table.removeRow(r)

    def update_log(self, message):
        self.log_console.append(message)

    def process_finished(self, message):
        self.update_log(message)
        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled):
        self.convert_btn.setEnabled(enabled)
        self.render_to_wav_btn.setEnabled(enabled)

    def browse_mp3_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.mp3 *.wav *.flac)")
        if file_path:
            self.mp3_path_edit.setText(file_path)

    def browse_midi_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择MIDI文件", "", "MIDI 文件 (*.mid *.midi)")
        if file_path:
            self.midi_path_edit.setText(file_path)

    def browse_midi_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择包含MIDI文件的文件夹")
        if dir_path:
            self.midi_path_edit.setText(dir_path)

    def run_process_dispatcher(self):
        input_path = self.mp3_path_edit.text()
        if not input_path or not os.path.exists(input_path):
            self.update_log("❌ 错误: 请先选择一个有效的音频输入文件。")
            return
        self.log_console.clear()
        self.set_buttons_enabled(False)
        if self.mode_stem_radio.isChecked():
            self._run_stem_mode_process(input_path)
        else:
            self._run_eq_mode_process(input_path)

    def _run_stem_mode_process(self, input_path):
        self.update_log("🚀 工作流开始：音轨分离模式")
        model_name = "htdemucs_ft"
        stems_dir = os.path.join(os.path.dirname(input_path), "separated_demucs", model_name)
        stem_paths = {k: os.path.join(stems_dir, f"{k}.wav") for k in ['vocals', 'drums', 'bass', 'other']}
        cache_exists = all(os.path.exists(p) for p in stem_paths.values())
        force_rerun = self.force_rerun_checkbox.isChecked()

        if cache_exists and not force_rerun:
            self.update_log("✅ 检测到已分离的音轨缓存。跳过分离步骤。")
            self.on_separation_finished(stem_paths)
        else:
            if force_rerun:
                self.update_log("ⓘ 用户选择强制重新分离。")
            else:
                self.update_log("ⓘ 未找到音轨缓存，开始完整分离流程。")
            self.update_log("第1步 - Demucs音轨分离")
            self.thread = QThread()
            self.worker = DemucsWorker()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(lambda: self.worker.run(input_path))
            self.worker.progress.connect(self.update_log)
            self.worker.finished.connect(self.on_separation_finished)
            self.worker.error.connect(self.process_finished)
            self.worker.error.connect(self.thread.quit)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()

    def on_separation_finished(self, stem_paths):
        self.update_log("✅ 音轨分离完成。")
        self.update_log("第2步 - 开始多音轨到MIDI的转换")
        base_name = os.path.splitext(os.path.basename(self.mp3_path_edit.text()))[0]
        output_folder = os.path.join(os.path.dirname(self.mp3_path_edit.text()), f"{base_name}_midi_output")
        midi_params = {
            'threshold_db': self.sensitivity_spin.value(),
            'frame_ms': self.time_precision_spin.value(),
            'conversion_mode': 'continuous' if self.note_mode_continuous.isChecked() else 'snapshot'
        }
        gains_db = {k: ctrl['spinbox'].value() for k, ctrl in self.stem_controls.items()}
        generate_spec = self.generate_spec_checkbox.isChecked()

        self.thread_midi = QThread()
        self.worker_midi = BatchedMidiConversionWorker()
        self.worker_midi.moveToThread(self.thread_midi)
        self.thread_midi.started.connect(
            lambda: self.worker_midi.run(stem_paths, midi_params, gains_db, output_folder, generate_spec))
        self.worker_midi.progress.connect(self.update_log)
        self.worker_midi.finished.connect(self.process_finished)
        self.worker_midi.error.connect(self.process_finished)
        self.worker_midi.finished.connect(self.thread_midi.quit)
        self.worker_midi.error.connect(self.thread_midi.quit)
        self.worker_midi.finished.connect(self.worker_midi.deleteLater)
        self.thread_midi.finished.connect(self.thread_midi.deleteLater)
        self.thread_midi.start()

    def _run_eq_mode_process(self, input_path):
        self.update_log("🚀 工作流开始:整体EQ模式")
        self.update_log("第1步 - 应用EQ")
        try:
            y, sr = librosa.load(input_path, sr=None)
            eq_bands = {}
            for row in range(self.eq_table.rowCount()):
                try:
                    low_hz = float(self.eq_table.item(row, 0).text())
                    high_hz = float(self.eq_table.item(row, 1).text())
                    gain_db = float(self.eq_table.item(row, 2).text())
                    eq_bands[(low_hz, high_hz)] = gain_db
                except (ValueError, AttributeError):
                    self.update_log(f"⚠️ 警告: EQ频段第 {row + 1} 行包含无效数值，已跳过。")
                    continue
            if eq_bands:
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                freqs = librosa.fft_frequencies(sr=sr)
                for (low_hz, high_hz), gain_db in eq_bands.items():
                    band_indices = np.where((freqs >= low_hz) & (freqs < high_hz))[0]
                    if len(band_indices) > 0:
                        S_db[band_indices, :] += gain_db
                S_amp = librosa.db_to_amplitude(S_db)
                y = librosa.istft(D / (np.abs(D) + 1e-8) * S_amp)
            self.update_log("✅ EQ应用完成。第2步 - MIDI转换")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_eq.mid")
            params = {
                'output_path': output_path,
                'threshold_db': self.sensitivity_spin.value(),
                'frame_ms': self.time_precision_spin.value(),
                'conversion_mode': 'continuous' if self.note_mode_continuous.isChecked() else 'snapshot'
            }
            self.thread_midi = QThread()
            self.worker_midi = SingleMidiGenerationWorker()
            self.worker_midi.moveToThread(self.thread_midi)
            self.thread_midi.started.connect(lambda: self.worker_midi.run(y, sr, params))
            self.worker_midi.progress.connect(self.update_log)
            self.worker_midi.finished.connect(self.process_finished)
            self.worker_midi.finished.connect(self.thread_midi.quit)
            self.worker_midi.finished.connect(self.worker_midi.deleteLater)
            self.thread_midi.finished.connect(self.thread_midi.deleteLater)
            self.thread_midi.start()
        except Exception as e:
            self.process_finished(f"❌ EQ处理步骤发生错误: {e}")

    def run_wav_rendering(self):
        input_path = self.midi_path_edit.text()
        if not input_path or not os.path.exists(input_path):
            self.update_log("❌ 错误: 请先选择一个有效的MIDI输入文件或文件夹。")
            return

        self.log_console.clear()
        self.set_buttons_enabled(False)
        tasks = []
        if os.path.isfile(input_path):
            self.update_log("🚀 检测到单个文件，开始渲染...")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_rendered.wav")
            tasks.append({'midi_path': input_path, 'output_path': output_path})
        elif os.path.isdir(input_path):
            self.update_log(f"📁 检测到文件夹，开始扫描MIDI文件...")
            for filename in os.listdir(input_path):
                if filename.lower().endswith(('.mid', '.midi')):
                    file_path = os.path.join(input_path, filename)
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(input_path, f"{base_name}_rendered.wav")
                    tasks.append({'midi_path': file_path, 'output_path': output_path})
            if not tasks:
                self.process_finished(f"🤷‍ 在文件夹 '{input_path}' 中没有找到任何MIDI文件。")
                return
        else:
            self.process_finished(f"❌ 错误: 输入路径 '{input_path}' 不是一个有效的文件或文件夹。")
            return

        params = {
            'tasks': tasks,
            'attack_ms': self.attack_spin.value(),
            'release_ms': self.release_spin.value(),
            'polyphony': self.polyphony_spin.value()
        }
        self.thread_wav = QThread()
        self.worker_wav = WavRenderingWorker()
        self.worker_wav.moveToThread(self.thread_wav)
        self.thread_wav.started.connect(lambda: self.worker_wav.run(params))
        self.worker_wav.progress.connect(self.update_log)
        self.worker_wav.finished.connect(self.process_finished)
        self.thread_wav.finished.connect(self.thread_wav.quit)
        self.worker_wav.finished.connect(self.worker_wav.deleteLater)
        self.thread_wav.finished.connect(self.thread_wav.deleteLater)
        self.thread_wav.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())