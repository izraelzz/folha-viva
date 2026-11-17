import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import '../models/plant_analysis.dart';

class PlantApiException implements Exception {
  const PlantApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class PlantApi {
  PlantApi({http.Client? client, String? baseUrl})
    : _client = client ?? http.Client(),
      _baseUrl =
          baseUrl ??
          const String.fromEnvironment(
            'API_BASE_URL',
            defaultValue: 'http://10.0.2.2:8000',
          );

  final http.Client _client;
  final String _baseUrl;

  Future<PlantAnalysis> analyze(XFile image, {String? crop}) async {
    try {
      final endpoint = Uri.parse(
        '${_baseUrl.replaceFirst(RegExp(r'/$'), '')}/analisar',
      );
      final request = http.MultipartRequest('POST', endpoint)
        ..files.add(await http.MultipartFile.fromPath('imagem', image.path));
      if (crop != null) request.fields['planta'] = crop;
      final streamedResponse = await _client
          .send(request)
          .timeout(const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        var message = 'A análise falhou (${response.statusCode}).';
        try {
          final body = jsonDecode(response.body);
          if (body is Map && body['detail'] is String) {
            message = body['detail'] as String;
          }
        } on FormatException {
          // Keep the HTTP status message when the server does not return JSON.
        }
        throw PlantApiException(message);
      }

      final body = jsonDecode(response.body);
      if (body is! Map<String, dynamic>) {
        throw const FormatException('A API retornou um formato inesperado.');
      }
      return PlantAnalysis.fromJson(body);
    } on TimeoutException {
      throw const PlantApiException(
        'A análise demorou demais. Verifique a conexão com a API e tente novamente.',
      );
    } on FileSystemException {
      throw const PlantApiException(
        'Não foi possível acessar a foto selecionada. Escolha a imagem novamente.',
      );
    } on SocketException {
      throw const PlantApiException(
        'Não foi possível conectar à API. Verifique se o servidor está ligado e acessível.',
      );
    } on http.ClientException {
      throw const PlantApiException(
        'Não foi possível conectar à API. Verifique se o servidor está ligado e acessível.',
      );
    } on FormatException {
      throw const PlantApiException(
        'A API retornou uma resposta inválida. Tente novamente.',
      );
    }
  }

  void close() => _client.close();
}
