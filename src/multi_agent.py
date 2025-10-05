import pandas as pd
import numpy as np
from collections import Counter
import re

class ResearchAgent:
    """base class for specialized research agents"""
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty
    
    def analyze(self, df):
        """override in subclass"""
        raise NotImplementedError

class TrendAnalystAgent(ResearchAgent):
    """analyzes research trends over time"""
    def __init__(self):
        super().__init__("Trend Analyst", "temporal patterns")
    
    def analyze(self, df):
        insights = []
        
        # analyze year distribution
        year_counts = df['year'].value_counts().sort_index()
        
        if len(year_counts) > 1:
            # find growth periods
            years = year_counts.index.tolist()
            counts = year_counts.values.tolist()
            
            # find largest growth
            max_growth = 0
            max_growth_period = None
            for i in range(len(years)-1):
                growth = counts[i+1] - counts[i]
                if growth > max_growth:
                    max_growth = growth
                    max_growth_period = (years[i], years[i+1])
            
            if max_growth_period:
                insights.append(
                    f"📈 major research spike from {max_growth_period[0]} to {max_growth_period[1]} "
                    f"with {max_growth} more publications"
                )
            
            # recent trend
            recent_years = years[-3:] if len(years) >= 3 else years
            recent_avg = sum([year_counts[y] for y in recent_years]) / len(recent_years)
            older_years = years[:-3] if len(years) >= 3 else []
            
            if older_years:
                older_avg = sum([year_counts[y] for y in older_years]) / len(older_years)
                if recent_avg > older_avg * 1.2:
                    insights.append("🚀 research activity accelerating in recent years")
                elif recent_avg < older_avg * 0.8:
                    insights.append("📉 research activity declining recently")
        
        return insights

class TopicExpertAgent(ResearchAgent):
    """identifies dominant topics and gaps"""
    def __init__(self):
        super().__init__("Topic Expert", "research domains")
    
    def analyze(self, df):
        insights = []
        
        # extract keywords from titles
        all_text = ' '.join(df['Title'].tolist()).lower()
        
        # hardcoded keywords based on domain expertise - alternative would be topic modeling but less interpretable
        keywords = {
            'microgravity': ['microgravity', 'weightlessness', 'zero-g'],
            'radiation': ['radiation', 'cosmic', 'ionizing'],
            'bone': ['bone', 'skeletal', 'osteo'],
            'muscle': ['muscle', 'muscular', 'myofiber'],
            'plant': ['plant', 'vegetation', 'botanical'],
            'immune': ['immune', 'immunity', 'immunological'],
            'cardiovascular': ['cardiovascular', 'heart', 'vascular'],
            'cellular': ['cell', 'cellular', 'cytology']
        }
        
        topic_counts = {}
        for topic, terms in keywords.items():
            count = sum([all_text.count(term) for term in terms])
            topic_counts[topic] = count
        
        # sort by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        # dominant topics
        if sorted_topics[0][1] > 0:
            insights.append(
                f"🔬 most researched area: {sorted_topics[0][0]} "
                f"({sorted_topics[0][1]} mentions)"
            )
        
        # underrepresented topics
        # magic number 10 - low enough to catch actual gaps but high enough to avoid noise
        low_topics = [t for t, c in sorted_topics if c < 10]
        if low_topics:
            insights.append(
                f"🔍 research gaps identified in: {', '.join(low_topics[:3])}"
            )
        
        # balance analysis
        # 5x ratio chosen as significant imbalance - could be dataset dependent
        top_count = sorted_topics[0][1]
        bottom_count = sorted_topics[-1][1]
        if bottom_count > 0:
            imbalance = top_count / bottom_count
            if imbalance > 5:
                insights.append(
                    f"⚠️ significant research imbalance detected - "
                    f"top topic has {imbalance:.1f}x more coverage"
                )
        
        return insights

class MethodologyAgent(ResearchAgent):
    """analyzes research methodologies and approaches"""
    def __init__(self):
        super().__init__("Methodology Analyst", "research methods")
    
    def analyze(self, df):
        insights = []
        
        # fallback to empty string for missing abstract column - common in research datasets
        all_text = ' '.join((df['Title'] + ' ' + df.get('abstract', '')).tolist()).lower()
        
        methods = {
            'in_vitro': ['in vitro', 'cell culture', 'cultured'],
            'in_vivo': ['in vivo', 'animal model', 'mice', 'rats'],
            'computational': ['simulation', 'computational', 'modeling', 'model'],
            'clinical': ['clinical', 'human', 'astronaut', 'crew'],
            'omics': ['genomic', 'proteomic', 'transcriptomic', 'metabolomic']
        }
        
        method_counts = {}
        for method, terms in methods.items():
            count = sum([all_text.count(term) for term in terms])
            method_counts[method] = count
        
        sorted_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_methods[0][1] > 0:
            insights.append(
                f"🧪 predominant methodology: {sorted_methods[0][0].replace('_', ' ')} studies"
            )
        
        # threshold of 5 mentions and 3 methods - heuristic for meaningful methodological diversity
        multi_method = sum([1 for c in method_counts.values() if c > 5])
        if multi_method >= 3:
            insights.append(
                f"✅ diverse methodological approaches detected ({multi_method} different methods)"
            )
        else:
            insights.append(
                "💡 opportunity for more methodological diversity"
            )
        
        return insights

class ImpactAnalystAgent(ResearchAgent):
    """analyzes potential impact and applications"""
    def __init__(self):
        super().__init__("Impact Analyst", "mission relevance")
    
    def analyze(self, df):
        insights = []
        
        all_text = ' '.join(df['Title'].tolist()).lower()
        
        missions = {
            'mars': ['mars', 'martian', 'red planet'],
            'moon': ['moon', 'lunar', 'artemis'],
            'iss': ['iss', 'international space station', 'station'],
            'deep_space': ['deep space', 'interplanetary', 'beyond leo']
        }
        
        mission_counts = {}
        for mission, terms in missions.items():
            count = sum([all_text.count(term) for term in terms])
            mission_counts[mission] = count
        
        total_mission_refs = sum(mission_counts.values())
        
        if total_mission_refs > 0:
            top_mission = max(mission_counts.items(), key=lambda x: x[1])
            insights.append(
                f"🎯 primary mission focus: {top_mission[0].replace('_', ' ')} "
                f"({top_mission[1]} references)"
            )
        
        countermeasures = ['countermeasure', 'mitigation', 'prevention', 'treatment', 'intervention']
        cm_count = sum([all_text.count(term) for term in countermeasures])
        
        # 20 mentions threshold - empirically chosen to distinguish focused vs incidental coverage
        if cm_count > 20:
            insights.append(
                f"💊 strong focus on health countermeasures ({cm_count} mentions)"
            )
        
        earth_apps = ['earth application', 'terrestrial', 'clinical application', 'medical']
        earth_count = sum([all_text.count(term) for term in earth_apps])
        
        if earth_count > 10:
            insights.append(
                "🌍 significant emphasis on earth applications and dual-use benefits"
            )
        
        return insights

class MultiAgentSystem:
    """orchestrates multiple specialized agents"""
    def __init__(self, df):
        self.df = df
        self.agents = [
            TrendAnalystAgent(),
            TopicExpertAgent(),
            MethodologyAgent(),
            ImpactAnalystAgent()
        ]
    
    def analyze(self):
        """run all agents and aggregate insights"""
        all_insights = []
        
        print("running multi-agent analysis...")
        
        for agent in self.agents:
            print(f"  - {agent.name} analyzing...")
            insights = agent.analyze(self.df)
            
            # add agent attribution
            for insight in insights:
                all_insights.append(f"**{agent.name}**: {insight}")
        
        # add meta-analysis
        all_insights.append(
            f"\n**System Summary**: analyzed {len(self.df)} publications using "
            f"{len(self.agents)} specialized agents"
        )
        
        return all_insights