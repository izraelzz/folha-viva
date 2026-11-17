import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../models/plant_analysis.dart';
import '../services/plant_api.dart';

const _green = Color(0xFF245B42);
const _ink = Color(0xFF1F3028);
const _muted = Color(0xFF748078);
const _line = Color(0xFFE5EAE4);
const _supportedCrops = ['Pepino', 'Melão-amargo'];

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  final _api = PlantApi();

  XFile? _image;
  PlantAnalysis? _analysis;
  String? _selectedCrop;
  bool _isAnalyzing = false;
  String? _error;

  @override
  void dispose() {
    _api.close();
    super.dispose();
  }

  Future<void> _selectImage(ImageSource source) async {
    try {
      final image = await _picker.pickImage(
        source: source,
        imageQuality: 88,
        maxWidth: 1800,
      );
      if (image == null || !mounted) return;

      setState(() {
        _image = image;
        _analysis = null;
        _error = null;
      });
    } on PlatformException {
      if (!mounted) return;
      setState(() {
        _error = 'Não foi possível abrir a câmera ou a galeria.';
      });
    }
  }

  Future<void> _analyze() async {
    final image = _image;
    final crop = _selectedCrop;
    if (image == null || crop == null || _isAnalyzing) return;

    setState(() {
      _isAnalyzing = true;
      _error = null;
      _analysis = null;
    });

    try {
      final analysis = await _api.analyze(image, crop: crop);
      if (!mounted) return;
      setState(() => _analysis = analysis);
    } on PlantApiException catch (error) {
      if (!mounted) return;
      setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _isAnalyzing = false);
    }
  }

  void _showImageOptions() {
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      backgroundColor: Colors.white,
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _ImageOption(
                icon: Icons.camera_alt_outlined,
                title: 'Tirar uma foto',
                onTap: () {
                  Navigator.pop(context);
                  _selectImage(ImageSource.camera);
                },
              ),
              _ImageOption(
                icon: Icons.photo_library_outlined,
                title: 'Escolher da galeria',
                onTap: () {
                  Navigator.pop(context);
                  _selectImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 22,
        title: const Row(
          children: [
            Icon(Icons.eco_rounded, color: _green, size: 25),
            SizedBox(width: 8),
            Text(
              'folha viva',
              style: TextStyle(
                color: _ink,
                fontSize: 20,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.5,
              ),
            ),
          ],
        ),
      ),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(22, 12, 22, 32),
          children: [
            const Text(
              'Cuidado que começa\ncom um olhar.',
              style: TextStyle(
                color: _ink,
                fontSize: 31,
                height: 1.12,
                fontWeight: FontWeight.w800,
                letterSpacing: -1,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'Fotografe uma folha e descubra como ela está.',
              style: TextStyle(color: _muted, fontSize: 15, height: 1.4),
            ),
            const SizedBox(height: 24),
            _photoCard(),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              initialValue: _selectedCrop,
              hint: const Text('Selecione Pepino ou Melão-amargo'),
              decoration: const InputDecoration(
                labelText: 'Cultura da planta',
                border: OutlineInputBorder(),
              ),
              items: _supportedCrops
                  .map(
                    (crop) => DropdownMenuItem(value: crop, child: Text(crop)),
                  )
                  .toList(),
              onChanged: _isAnalyzing
                  ? null
                  : (crop) {
                      if (crop == null) return;
                      setState(() {
                        _selectedCrop = crop;
                        _analysis = null;
                        _error = null;
                      });
                    },
            ),
            const SizedBox(height: 8),
            const Text(
              'Selecione a cultura da folha. Se o modelo discordar da sua escolha, o resultado será marcado como incerto.',
              style: TextStyle(color: _muted, fontSize: 12, height: 1.4),
            ),
            const SizedBox(height: 16),
            if (_image == null)
              ElevatedButton.icon(
                onPressed: _showImageOptions,
                icon: const Icon(Icons.add_a_photo_outlined),
                label: const Text('Escolher uma foto'),
              )
            else
              ElevatedButton.icon(
                onPressed: _isAnalyzing || _selectedCrop == null
                    ? null
                    : _analyze,
                icon: _isAnalyzing
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.center_focus_strong_outlined),
                label: Text(_isAnalyzing ? 'Analisando...' : 'Analisar planta'),
              ),
            if (_image != null) ...[
              const SizedBox(height: 10),
              TextButton.icon(
                onPressed: _isAnalyzing ? null : _showImageOptions,
                icon: const Icon(Icons.refresh_rounded, size: 18),
                label: const Text('Escolher outra foto'),
                style: TextButton.styleFrom(foregroundColor: _green),
              ),
            ],
            if (_error != null) ...[
              const SizedBox(height: 12),
              _ErrorMessage(message: _error!),
            ],
            if (_analysis != null) ...[
              const SizedBox(height: 22),
              _ResultCard(analysis: _analysis!),
            ],
            const SizedBox(height: 22),
            const _InfoNote(),
          ],
        ),
      ),
    );
  }

  Widget _photoCard() {
    return Container(
      height: 240,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: const Color(0xFFEAF0E8),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _line),
      ),
      child: _image == null
          ? InkWell(
              onTap: _showImageOptions,
              child: const Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircleAvatar(
                    radius: 31,
                    backgroundColor: Colors.white,
                    child: Icon(
                      Icons.add_a_photo_outlined,
                      color: _green,
                      size: 27,
                    ),
                  ),
                  SizedBox(height: 14),
                  Text(
                    'Sua planta começa aqui',
                    style: TextStyle(
                      color: _ink,
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    'Toque para fotografar ou enviar uma imagem',
                    style: TextStyle(color: _muted, fontSize: 12),
                  ),
                ],
              ),
            )
          : Stack(
              fit: StackFit.expand,
              children: [
                Image.file(File(_image!.path), fit: BoxFit.cover),
                Positioned(
                  right: 12,
                  top: 12,
                  child: IconButton.filled(
                    onPressed: _isAnalyzing ? null : _showImageOptions,
                    tooltip: 'Trocar foto',
                    style: IconButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: _green,
                    ),
                    icon: const Icon(Icons.edit_outlined),
                  ),
                ),
              ],
            ),
    );
  }
}

class _ImageOption extends StatelessWidget {
  const _ImageOption({
    required this.icon,
    required this.title,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Icon(icon, color: _green),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
      onTap: onTap,
    );
  }
}

class _ErrorMessage extends StatelessWidget {
  const _ErrorMessage({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF1EF),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline, color: Color(0xFFB63F32), size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(color: Color(0xFF8A332B), height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.analysis});

  final PlantAnalysis analysis;

  @override
  Widget build(BuildContext context) {
    final healthy = analysis.isHealthy;
    final color = analysis.isUncertain
        ? _muted
        : healthy
        ? _green
        : const Color(0xFFB66A28);
    final title = analysis.isUncertain
        ? 'Resultado incerto'
        : healthy
        ? 'Folha saudável'
        : 'Possível alteração';

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                radius: 23,
                backgroundColor: color.withValues(alpha: 0.1),
                child: Icon(
                  healthy ? Icons.check_rounded : Icons.spa_outlined,
                  color: color,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: TextStyle(
                        color: color,
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    if (!analysis.cropMismatch) ...[
                      const SizedBox(height: 3),
                      Text(
                        'Pontuação do modelo: ${(analysis.confidence * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(color: _muted, fontSize: 13),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            analysis.message,
            style: const TextStyle(color: _ink, height: 1.4),
          ),
          if (analysis.alternatives.isNotEmpty) ...[
            const SizedBox(height: 17),
            const Text(
              'Possibilidades',
              style: TextStyle(color: _ink, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 10),
            for (final option in analysis.alternatives)
              _AlternativeRow(option: option),
          ],
          const SizedBox(height: 16),
          const Divider(color: _line, height: 1),
          const SizedBox(height: 14),
          const Text(
            'Cuidados gerais',
            style: TextStyle(color: _ink, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(
            analysis.recommendation,
            style: const TextStyle(color: _muted, height: 1.45),
          ),
          const SizedBox(height: 12),
          Text(
            'Análise concluída em ${analysis.elapsedMilliseconds.toStringAsFixed(0)} ms',
            style: const TextStyle(color: _muted, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _AlternativeRow extends StatelessWidget {
  const _AlternativeRow({required this.option});

  final PlantAlternative option;

  @override
  Widget build(BuildContext context) {
    final percentage = (option.confidence * 100).clamp(0, 100).toDouble();
    return Padding(
      padding: const EdgeInsets.only(bottom: 11),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  option.name,
                  style: const TextStyle(color: _ink, fontSize: 13),
                ),
              ),
              Text(
                '${percentage.toStringAsFixed(1)}%',
                style: const TextStyle(
                  color: _muted,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 5),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: percentage / 100,
              minHeight: 5,
              backgroundColor: const Color(0xFFE9EEE8),
              color: option.healthy ? _green : const Color(0xFFB66A28),
            ),
          ),
        ],
      ),
    );
  }
}

class _InfoNote extends StatelessWidget {
  const _InfoNote();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: const Color(0xFFEDF2EA),
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline_rounded, color: _green, size: 20),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'Este app analisa somente pepino e melão-amargo. Outras plantas não são suportadas. As porcentagens são pontuações não calibradas; confirme suspeitas com assistência agronômica.',
              style: TextStyle(color: _ink, fontSize: 12, height: 1.45),
            ),
          ),
        ],
      ),
    );
  }
}
