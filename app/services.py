import random


class NLPService:
    @staticmethod
    def analyze_stress_level(text: str) -> tuple[str, int]:
        """
        පරිශීලකයාගේ text එක විශ්ලේෂණය කර පීඩන මට්ටම (0-100) සහ පිළිතුර ලබා දේ.
        """
        text_lower = text.lower()

        # High Stress Keywords (සිංහල සහ ඉංග්‍රීසි කලවම් සිංහල - Hinglish ද ඇතුළුව)
        high_stress_words = ["stress", "ස්ට්‍රෙස්", "බය", "අවුල්", "එපා වෙලා", "කේන්ති", "අඬන්න", "exam", "පීඩනය"]
        # Low Stress Keywords
        low_stress_words = ["හොඳයි", "සතුටු", "නියමයි", "happy", "පට්ට", "ela", "supිරි"]

        # 1. High Stress තත්ත්වයක් හඳුනා ගැනීම
        if any(word in text_lower for word in high_stress_words):
            stress_score = random.randint(71, 100)  # Sad State
            reply = (
                f"ඔයා කියපු දේ මට තේරුණා දිනුෂ. ඔයා ලොකු පීඩනයකින් ඉන්න බවක් පේනවා ({stress_score}% Stress). "
                f"අපි මුලින්ම Activities ටැබ් එකට ගිහින් පොඩි Breathing Exercise එකක් කරලා එමුද?"
            )
            return reply, stress_score

        # 2. Low Stress / Happy තත්ත්වයක් හඳුනා ගැනීම
        elif any(word in text_lower for word in low_stress_words):
            stress_score = random.randint(0, 30)  # Happy State
            reply = f"නියමයි! ඔයා ඉතාම සන්සුන් මනසකින් ඉන්න බවක් පේනවා ({stress_score}% Stress). ඔය සතුට දිගටම පවත්වා ගමු!"
            return reply, stress_score

        # 3. Neutral / සාමාන්‍ය තත්ත්වය
        else:
            stress_score = random.randint(31, 70)  # Neutral State
            reply = f"ඔයා කියපු දේ මට වැටහුණා. දැනට ඔයාගේ පීඩන මට්ටම සාමාන්‍ය අගයක තියෙන්නේ ({stress_score}% Stress). තව විස්තර කියන්න."
            return reply, stress_score