class PlantAnalysis {
  const PlantAnalysis({
    required this.result,
    required this.confidence,
    required this.message,
    required this.elapsedMilliseconds,
    required this.probabilities,
    required this.alternatives,
    required this.recommendation,
    required this.isUncertain,
    this.cropMismatch = false,
  });

  final String result;
  final double confidence;
  final String message;
  final double elapsedMilliseconds;
  final Map<String, double> probabilities;
  final List<PlantAlternative> alternatives;
  final String recommendation;
  final bool isUncertain;
  final bool cropMismatch;

  bool get isHealthy => alternatives.isNotEmpty
      ? alternatives.first.healthy
      : result == 'saudavel';

  factory PlantAnalysis.fromJson(Map<String, dynamic> json) {
    final rawProbabilities = json['probabilidades'];
    if (rawProbabilities is! Map) {
      throw const FormatException('Resposta de probabilidades inválida.');
    }
    final rawAlternatives = json['alternativas'];
    final alternatives = rawAlternatives is List
        ? rawAlternatives
              .map(
                (entry) => PlantAlternative.fromJson(
                  Map<String, dynamic>.from(entry as Map),
                ),
              )
              .toList()
        : <PlantAlternative>[];

    return PlantAnalysis(
      result: json['resultado'] as String,
      confidence: (json['confianca'] as num).toDouble(),
      message: json['mensagem'] as String,
      elapsedMilliseconds: (json['tempo_ms'] as num).toDouble(),
      probabilities: rawProbabilities.map(
        (key, value) => MapEntry(key.toString(), (value as num).toDouble()),
      ),
      alternatives: alternatives,
      recommendation:
          json['orientacao'] as String? ??
          'Confirme o resultado com um profissional antes de agir.',
      isUncertain: json['incerta'] as bool? ?? false,
      cropMismatch: json['cultura_incompativel'] as bool? ?? false,
    );
  }
}

class PlantAlternative {
  const PlantAlternative({
    required this.name,
    required this.plant,
    required this.condition,
    required this.confidence,
    required this.healthy,
  });

  final String name;
  final String plant;
  final String condition;
  final double confidence;
  final bool healthy;

  factory PlantAlternative.fromJson(Map<String, dynamic> json) {
    return PlantAlternative(
      name: json['classe'] as String,
      plant: json['planta'] as String,
      condition: json['condicao'] as String,
      confidence: (json['confianca'] as num).toDouble(),
      healthy: json['saudavel'] as bool,
    );
  }
}
