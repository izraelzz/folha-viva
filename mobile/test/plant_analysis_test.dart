import 'package:flutter_test/flutter_test.dart';

import 'package:detector_plantas_mobile/models/plant_analysis.dart';

void main() {
  test('parses the backend analysis response', () {
    final analysis = PlantAnalysis.fromJson({
      'resultado': 'saudavel',
      'confianca': 0.96,
      'mensagem': 'A folha parece saudável.',
      'tempo_ms': 12.5,
      'probabilidades': {'doente': 0.04, 'saudavel': 0.96},
      'alternativas': [
        {
          'classe': 'Tomate — Folha saudável',
          'planta': 'Tomate',
          'condicao': 'Folha saudável',
          'confianca': 0.96,
          'saudavel': true,
        },
      ],
      'orientacao': 'Continue monitorando a planta.',
      'incerta': false,
    });

    expect(analysis.isHealthy, isTrue);
    expect(analysis.confidence, 0.96);
    expect(analysis.probabilities['doente'], 0.04);
    expect(analysis.alternatives.single.condition, 'Folha saudável');
    expect(analysis.recommendation, 'Continue monitorando a planta.');
    expect(analysis.isUncertain, isFalse);
  });

  test('rejects a response without class probabilities', () {
    expect(
      () => PlantAnalysis.fromJson({
        'resultado': 'saudavel',
        'confianca': 0.96,
        'mensagem': 'A folha parece saudável.',
        'tempo_ms': 12.5,
      }),
      throwsFormatException,
    );
  });

  test('represents a crop mismatch without presenting a disease', () {
    final analysis = PlantAnalysis.fromJson({
      'resultado': 'incerto',
      'confianca': 0.0,
      'mensagem': 'Não foi possível analisar esta imagem como Cabaça. Nenhuma doença foi indicada.',
      'tempo_ms': 12.5,
      'probabilidades': <String, double>{},
      'alternativas': <Map<String, dynamic>>[],
      'orientacao': 'Confira a cultura.',
      'incerta': true,
      'cultura_incompativel': true,
    });

    expect(analysis.cropMismatch, isTrue);
    expect(analysis.isUncertain, isTrue);
    expect(analysis.alternatives, isEmpty);
    expect(analysis.confidence, 0);
  });
}
