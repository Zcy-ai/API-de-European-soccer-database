import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
st.title("Data Analysis on European Soccer Database from Kaggle")
st.write("By Zha Chenyi")
conn = sqlite3.connect('database——soccer.sqlite')
c=conn.cursor()
function_dict={'function':['The result of game','Prediction of one game of BBVA','Attributes of player','Attributes of team','Home and away win rate']}
df_function = pd.DataFrame(function_dict)
option_function = st.sidebar.selectbox('Which function do you want to select?',df_function)
############function 1: Let users can easily find the result of one match##############
if (option_function == 'The result of game'):
    query_league = '''
    SELECT name
    FROM league
    '''
    league_name=c.execute(query_league)
    df_league_name = pd.DataFrame(league_name,columns=['league_name'])
#use SQL to filter out the information of the match
    query_match = """
    SELECT m.match_api_id AS match_id, c.name AS country, l.name AS league, m.season, m.stage, m.date, home.team_long_name AS home_team,
       away.team_long_name AS away_team, m.home_team_goal AS home_goal, m.away_team_goal AS away_goal
    FROM match AS m, country AS c, league AS l, team home, team away
    WHERE m.country_id = c.id AND m.league_id = l.id and
      home.team_api_id = m.home_team_api_id AND away.team_api_id = m.away_team_api_id
    """
    c.execute(query_match)
    data_match=c.fetchall()
    columns = [desc[0] for desc in c.description]
#transform les informations to dataframe
    match_info=pd.DataFrame(data_match,columns=columns)
#Let users give their filter criteria 
    option_league = st.sidebar.selectbox('Game of which League do you want to research?',df_league_name)
    st.write('You selected:',option_league)
    dic_season = {'season':['2008/2009','2009/2010','2010/2011','2011/2012','2012/2013','2014/2015','2015/2016']}
    df_season = pd.DataFrame(dic_season)
    option_season = st.sidebar.selectbox('Game of which season do you want to research?',df_season)
    st.write('You selected:',option_season)
    option_home_team = st.sidebar.text_input('Game of which home team do you want to research?', 'FC Barcelona')
    option_away_team = st.sidebar.text_input('Game of which away team do you want to research?', 'Real Madrid CF')
#give out the final results
    match_info_1=match_info.loc[match_info["season"]==option_season]
    match_info_2=match_info_1.loc[match_info["league"]==option_league]
    match_info_3=match_info_2.loc[match_info["home_team"]==option_home_team]
    match_info_4=match_info_3.loc[match_info["away_team"]==option_away_team]
    st.write(match_info_4)
##########function 2: Give a prediction of one game of BBVA based on the results of several seasons###########
elif (option_function == 'Prediction of one game of BBVA'):
    st.header('The Prediction of two teams belonging to spain BBVA')
    option_home_team = st.sidebar.text_input('Which home team of BBVA do you want to select?','FC Barcelona')
    option_away_team = st.sidebar.text_input('Which away team of BBVA do you want to select?', 'Real Madrid CF')
    st.subheader("Home team:")
    st.write(option_home_team)
    st.subheader("VS")
    st.subheader("Away team:",option_away_team)
    st.write(option_away_team)
#sort out the game information and team attributes of the home team,because we have to take home advantage into account
    query_home = '''
    SELECT match.id,team.team_long_name AS home_team_name, home_team_goal ,away_team_goal , team_attributes.buildUpPlaySpeed AS home_buildupplayspeed,
    team_attributes.buildUpPlayPassing AS home_buildupplaypassing , team_attributes.chanceCreationPassing AS home_chancecreationpassing,
    team_attributes.chanceCreationCrossing AS home_chancecreationcrossing,team_attributes.chanceCreationShooting AS home_chancecreationshooting , 
    team_attributes.defencePressure AS home_defencepressure , team_attributes.defenceAggression AS home_defenceaggression , team_attributes.defenceTeamWidth AS home_defenceteamwidth
    FROM (match INNER JOIN team ON match.home_team_api_id = team.team_api_id) 
    INNER JOIN team_attributes ON match.home_team_api_id = team_attributes.team_api_id
    WHERE league_id=21518
    '''
    c.execute(query_home)
    data_home_team=c.fetchall()
    columns= [desc[0] for desc in c.description]
    result_info_home=pd.DataFrame(data_home_team,columns=columns)
#The team attributes are based on the season,so we need to give the average attribute of one team
    result_info_home=result_info_home.groupby('id').mean().reset_index()
#sort out the game information and team attributes of the away team
    query_away = '''
    SELECT match.id,team.team_long_name AS away_team_name, team_attributes.buildUpPlaySpeed AS away_buildupplayspeed,
    team_attributes.buildUpPlayPassing AS away_buildupplaypassing ,  team_attributes.chanceCreationPassing AS away_chancecreationpassing,
    team_attributes.chanceCreationCrossing AS away_chancecreationcrossing,team_attributes.chanceCreationShooting AS away_chancecreationshooting , 
    team_attributes.defencePressure AS away_defencepressure , team_attributes.defenceAggression AS away_defenceaggression ,  team_attributes.defenceTeamWidth AS away_defenceteamwidth
    FROM (match INNER JOIN team ON match.away_team_api_id = team.team_api_id) 
    INNER JOIN team_attributes ON match.away_team_api_id = team_attributes.team_api_id
    WHERE league_id =21518
    '''
    c.execute(query_away)
    data_away_team=c.fetchall()
    columns= [desc[0] for desc in c.description]
    result_info_away=pd.DataFrame(data_away_team,columns=columns)
#average attribute
    result_info_away=result_info_away.groupby('id').mean().reset_index()
#Combine match results,home team attributes and away team attributes into one table
    result_info=pd.merge(result_info_home,result_info_away)
    classe={}
    for i in range(len(result_info)):
        if(result_info.iloc[i,1]>result_info.iloc[i,2]):
            classe[i]=2
        elif(result_info.iloc[i,1]==result_info.iloc[i,2]):
            classe[i]=1
        elif(result_info.iloc[i,1]<result_info.iloc[i,2]):
            classe[i]=0
    classe_list=list(classe.values())
    result_info['class']=classe_list
    result_info=result_info.drop('id',axis=1)
    result_info=result_info.drop('home_team_goal',axis=1)
    result_info=result_info.drop('away_team_goal',axis=1)
#determine the training set and test set 
    x=result_info.drop('class',axis=1)
    y=result_info['class']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
#fit the model of SVC
    model = SVC()
    model.fit(x_train,y_train)
#Now we have to use the model to give the result of prediction 
    query_2='''
    SELECT team.team_long_name,team_attributes.buildUpPlaySpeed AS home_buildupplayspeed,team_attributes.buildUpPlayPassing AS home_buildupplaypassing , team_attributes.chanceCreationPassing AS home_chancecreationpassing,
    team_attributes.chanceCreationCrossing AS home_chancecreationcrossing,team_attributes.chanceCreationShooting AS home_chancecreationshooting , 
    team_attributes.defencePressure AS home_defencepressure , team_attributes.defenceAggression AS home_defenceaggression , team_attributes.defenceTeamWidth AS home_defenceteamwidth
    FROM team_attributes
    JOIN team ON team.team_api_id = team_attributes.team_api_id
    '''
    c.execute(query_2)
    data=c.fetchall()
    columns= [desc[0] for desc in c.description]
    home_team_attributes_info=pd.DataFrame(data,columns=columns)
    home_team_attributes_info=home_team_attributes_info.groupby('team_long_name').mean().reset_index()
    home_team_attributes_info=home_team_attributes_info.loc[home_team_attributes_info["team_long_name"]==option_home_team]
    #home_team_attributes_info=home_team_attributes_info.drop('team_long_name',axis=1)
    query_1='''
    SELECT 
    team.team_long_name,team_attributes.buildUpPlaySpeed AS away_buildupplayspeed,
    team_attributes.buildUpPlayPassing AS away_buildupplaypassing ,  team_attributes.chanceCreationPassing AS away_chancecreationpassing,
    team_attributes.chanceCreationCrossing AS away_chancecreationcrossing,team_attributes.chanceCreationShooting AS away_chancecreationshooting , 
    team_attributes.defencePressure AS away_defencepressure , team_attributes.defenceAggression AS away_defenceaggression ,  team_attributes.defenceTeamWidth AS away_defenceteamwidth
    FROM team_attributes
    JOIN team ON team.team_api_id = team_attributes.team_api_id
    '''
    c.execute(query_1)
    data=c.fetchall()
    columns= [desc[0] for desc in c.description]
    away_team_attributes_info=pd.DataFrame(data,columns=columns)
    away_team_attributes_info=away_team_attributes_info.groupby('team_long_name').mean().reset_index()
    away_team_attributes_info=away_team_attributes_info.loc[away_team_attributes_info["team_long_name"]==option_away_team]
    away_team_attributes_info=away_team_attributes_info.drop('team_long_name',axis=1)
    away_team_attributes_info['team_long_name']=option_home_team
    select_team_attributes=pd.merge(home_team_attributes_info,away_team_attributes_info)
    select_team_attributes=select_team_attributes.drop('team_long_name',axis=1)
    y_pred=model.predict(select_team_attributes)
    st.subheader("The result of the game is:")
    st.write(y_pred)
    if(y_pred == 2):
        st.subheader("We predict the home team win!")
    elif(y_pred ==1):
        st.subheader("We predict is a tie game!")
    elif(y_pred == 0):
        st.subheader("We predict the away team win")
##########function 3: Give some informations of player's attributes##########
elif (option_function == 'Attributes of player'):
    st.header("The table of attributes of the player you have chosen")
#select the player you want to see whose attribute
    FOOTBALL_PLAYER = st.text_input('Whose attributes you want to know?', 'Lionel Messi')
#query contain the attributes of each player
    query = """
    SELECT P.player_name,P.weight,P.height,PA.overall_rating, 
       PA.potential, PA.preferred_foot, PA.attacking_work_rate, PA.defensive_work_rate, PA.crossing, PA.finishing,
       PA.heading_accuracy, PA.short_passing, PA.volleys, PA.dribbling, PA.curve, PA.free_kick_accuracy,
       PA.long_passing, PA.ball_control, PA.acceleration, PA.sprint_speed, PA.agility, PA.reactions, PA.balance,
       PA.shot_power, PA.jumping, PA.stamina, PA.strength, PA.long_shots, PA.aggression, PA.interceptions,
       PA.positioning, PA.vision, PA.penalties, PA.marking, PA.standing_tackle, PA.sliding_tackle, PA.gk_diving,
       PA.gk_handling, PA.gk_kicking, PA.gk_positioning, PA.gk_reflexes 
    FROM player AS P
    JOIN player_attributes AS PA ON P.player_api_id = PA.player_api_id 
    """
    c.execute(query)
    data=c.fetchall()
    columns = [desc[0] for desc in c.description]
    player_info=pd.DataFrame(data,columns=columns)
#il y a 8 seasons des attributes de joueurs ,on doit prendre le mean de valeur des attributes, et reset_index nous permet de maintenir le formule dataframe
    player_info_mean=player_info.groupby('player_name').mean().reset_index()
    st.write(player_info_mean.loc[player_info_mean["player_name"] == FOOTBALL_PLAYER])
# plot the "The radar map"
    st.header("The Radar map of the attributes of player you have chosen")
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    feature = ['crossing','finishing', 'heading_accuracy', 'short_passing', 'dribbling', 'ball_control', 'shot_power', 'acceleration', 'stamina','long_shots', 'vision', 'penalties']
    player_info_chosen=player_info_mean.loc[player_info_mean["player_name"] == FOOTBALL_PLAYER]
#Pour acquerir les valeurs de l'attribute dont on a besoin
    value_1=player_info_chosen.iat[0,5]
    value_2=player_info_chosen.iat[0,6]
    value_3=player_info_chosen.iat[0,7]
    value_4=player_info_chosen.iat[0,8]
    value_5=player_info_chosen.iat[0,10]
    value_6=player_info_chosen.iat[0,14]
    value_7=player_info_chosen.iat[0,20]
    value_8=player_info_chosen.iat[0,15]
    value_9=player_info_chosen.iat[0,22]
    value_10=player_info_chosen.iat[0,24]
    value_11=player_info_chosen.iat[0,28]
    value_12=player_info_chosen.iat[0,29]
    values = [value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9,value_10,value_11,value_12]
    angles=np.linspace(0, 2*np.pi,len(values), endpoint=False)
#Stiching the end of the data to close the lines in the graphe
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    feature = np.concatenate((feature,[feature[0]]))
#time to plot
    fig=plt.figure()
#set to polar coordinates
    ax = fig.add_subplot(111, polar=True)
#draw a line chart
    ax.plot(angles, values, 'o-', linewidth=2)
#fill with color
    ax.fill(angles, values, alpha=0.25) 
#Set the angle division scale on the icon and add a label for each data point
    ax.set_thetagrids(angles * 180/np.pi, feature)
#set the limit of coordinates
    ax.set_ylim(0,100)
#give the title
    plt.title('The radar map of the chosen player')
#add grid line
    ax.grid(True) 
    plt.show()
    st.pyplot(fig)
    st.header('Comparision of two players')
    FOOTBALL_PLAYER_compare = st.text_input('Who you want to know compare with?', 'Cristiano Ronaldo')
#select un joueur pour comparer,merge two line charts
    player_info_chosen_2=player_info_mean.loc[player_info_mean["player_name"] == FOOTBALL_PLAYER_compare]
    value_1=player_info_chosen_2.iat[0,5]
    value_2=player_info_chosen_2.iat[0,6]
    value_3=player_info_chosen_2.iat[0,7]
    value_4=player_info_chosen_2.iat[0,8]
    value_5=player_info_chosen_2.iat[0,10]
    value_6=player_info_chosen_2.iat[0,14]
    value_7=player_info_chosen_2.iat[0,20]
    value_8=player_info_chosen_2.iat[0,15]
    value_9=player_info_chosen_2.iat[0,22]
    value_10=player_info_chosen_2.iat[0,24]
    value_11=player_info_chosen_2.iat[0,28]
    value_12=player_info_chosen_2.iat[0,29]
    values_compare=[value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9,value_10,value_11,value_12]
    values_compare=np.concatenate((values_compare,[values_compare[0]]))
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2,label=FOOTBALL_PLAYER)
    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values_compare, 'o-', linewidth=2,label=FOOTBALL_PLAYER_compare)
    ax.fill(angles, values_compare, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, feature)
    ax.set_ylim(0,100)
    plt.title('The radar map of the comparision of two players')
    plt.legend(loc='best')
    ax.grid(True)
    plt.show()
    st.pyplot(fig)
    st.header("Top 10 players with the top over_rating")
    st.write(player_info.groupby('player_name').mean().sort_values('overall_rating', ascending=False)[:10])
####
    av_best10_potential = (player_info.groupby('player_name').mean().sort_values('potential', ascending=False)[:10].mean())
    av_best10_rating = (player_info.groupby('player_name').mean().sort_values('overall_rating', ascending=False)[:10].mean())
    average_pl = player_info.mean(numeric_only=True)
    fig0 = plt.figure(figsize = (20, 30)) 
    ax0 = fig0.add_subplot()
    ind0 = np.arange(len(average_pl))
    plt.barh(ind0+0.25, av_best10_potential, color ='red', height = 0.25, label='Top 10 Potenial Players')
    plt.barh(ind0, av_best10_rating, color ='blue', height = 0.25, label='Top 10 Overall Rating Players')
    plt.barh(ind0-0.25, average_pl, color ='orange', height = 0.25, label='Average Attributes')
    ax0.set_yticks(ind0)
    ax0.set_yticklabels(average_pl.index.tolist())
    plt.yticks(fontsize=15)
    plt.ylabel("Player Attributes", fontsize=18)
    plt.xlim(0,100)
    plt.xlabel("Percent", fontsize=18)
    plt.title("Comparing Player Attributes of The Top Rated Players to Attributes of Other Players", fontsize=23)
    plt.legend()
    plt.show()
    st.pyplot(fig0)
#heatmap of each attributes of player
    st.header("The heatmap of correlation of the player attribute")
    heat_map=plt.figure(figsize=(25,20))
    player_attribute=player_info_mean.drop('player_name',axis=1)
    sns.heatmap(player_attribute.corr(method='pearson'), annot=True, cmap="YlGnBu")
    st.write(heat_map)
##########function 4: Give some informations of team's attributes##########
elif (option_function == 'Attributes of team'):
    st.header("The attributes of your chosen team")
    FOOTBALL_TEAM = st.sidebar.text_input('Attributes of which team you want to know?', 'FC Barcelona')
#The information of each team with different seasons
    query ='''
    SELECT 
    team.team_long_name,team_attributes.buildUpPlaySpeed , team_attributes.buildUpPlayPassing , team_attributes.chanceCreationPassing    ,team_attributes.chanceCreationCrossing   ,team_attributes.chanceCreationShooting , team_attributes.defencePressure, team_attributes.defenceAggression, team_attributes.defenceTeamWidth
    FROM team 
    JOIN team_attributes ON team.team_api_id = team_attributes.team_api_id
    '''
    c.execute(query)
    data=c.fetchall()
    columns= [desc[0] for desc in c.description]
    team_attributes_info=pd.DataFrame(data,columns=columns)
    team_attributes_info=team_attributes_info.groupby('team_long_name').mean().reset_index()
    team_attributes_info_select=team_attributes_info.loc[team_attributes_info["team_long_name"]==FOOTBALL_TEAM]
    st.write(team_attributes_info_select)
#plot a heatmap for knowing the correlation of different attributes of team
    st.header("The heatmap of correlation of the team attribute")
    heat_map=plt.figure(figsize=(20,16))
    team_attributes_info=team_attributes_info.drop('team_long_name',axis=1)
    sns.heatmap(team_attributes_info.corr(method='pearson'), annot=True, cmap="YlGnBu")
    st.write(heat_map)
##########function 5: Give some informations of home team win rate##########
elif (option_function == 'Home and away win rate'):
    st.header("table of the win rate of home team")
    query_league = '''
    SELECT name
    FROM league
    '''
    league_name=c.execute(query_league)
    df_league_name = pd.DataFrame(league_name,columns=['league_name'])
    dic_season = {'season':['2008/2009','2009/2010','2010/2011','2011/2012','2012/2013','2014/2015','2015/2016']}
    df_season = pd.DataFrame(dic_season)
    league_option=st.selectbox('home team win rate of which league?',df_league_name)
    season_option=st.selectbox('home team win rate of which season?',df_season)
#count the number of wins for the home team
    query = '''
    SELECT match.season,league.name AS league ,COUNT(*) AS num_win
    FROM match 
    JOIN league ON match.league_id =league.id
    WHERE match.home_team_goal >= match.away_team_goal
    GROUP BY league.name,season
    '''
    c.execute(query)
    data_1=c.fetchall()
    columns= [desc[0] for desc in c.description]
    win_rate_info=pd.DataFrame(data_1,columns=columns)
#The total number of matches
    query = '''
    SELECT match.season,league.name,COUNT(*) as Count
    FROM match 
    JOIN league ON match.league_id =league.id
    GROUP BY league.name,season
    '''
    c.execute(query)
    data_2=c.fetchall()
    columns = [desc[0] for desc in c.description]
    num_game=pd.DataFrame(data_2,columns=columns)
#give the table of win_rate of home team
    win_rate_info["num_game"]=num_game["Count"]
    win_rate_info["win_rate"]=win_rate_info["num_win"]/win_rate_info["num_game"]    
    win_rate_info_1=win_rate_info.loc[win_rate_info["season"] == season_option]
    win_rate_info_2=win_rate_info_1.loc[win_rate_info["league"] == league_option]
    st.write(win_rate_info_2)
#give the  bar chart of win rate of home team by league in different seasons,give altair a try
    st.header("plot of the win rate of home team per season")
    season_option_1=st.selectbox('plot of the win rate of home team for which season?',df_season)
    df_season=win_rate_info.loc[win_rate_info["season"] == season_option_1]
    df_1=df_season[['league','win_rate']]
    st.subheader('win rate of home team of season you choose')
    c_1=alt.Chart(df_1).mark_bar().encode(x='league',y='win_rate').properties(width=600,height=500)
    st.altair_chart(c_1)
#lineplot
    st.subheader('Win rate of home team changes with the season')
    league_option_1=st.selectbox('plot of the the teandance of the win rate of home team for which league?',df_league_name)
    win_rate_select=win_rate_info.loc[win_rate_info["league"]==league_option_1]
    fig=plt.figure(figsize=(20,16))
    sns.lineplot(x="season", y="win_rate", data=win_rate_select)
    st.write(fig)