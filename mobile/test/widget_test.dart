import 'package:flutter_test/flutter_test.dart';

import 'package:detector_plantas_mobile/main.dart';

void main() {
  testWidgets('shows the plant photo analysis home screen', (tester) async {
    await tester.pumpWidget(const PlantaVivaApp());

    expect(find.text('folha viva'), findsOneWidget);
    expect(find.text('Cuidado que começa\ncom um olhar.'), findsOneWidget);
    expect(find.text('Cultura da planta'), findsOneWidget);
    expect(find.text('Escolher uma foto'), findsOneWidget);
  });
}
