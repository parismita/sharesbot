using System;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;
using System.Web.Http.Description;
using Microsoft.Bot.Connector;
using QC = System.Data.SqlClient;
using Newtonsoft.Json;

namespace stockbot
{
    [BotAuthentication]
    public class MessagesController : ApiController
    {
        /// <summary>
        /// POST: api/Messages
        /// Receive a message from a user and reply to it
        /// </summary>
        public async Task<HttpResponseMessage> Post([FromBody]Activity activity)
        {
            if (activity.Type == ActivityTypes.Message)
            {
                ConnectorClient connector = new ConnectorClient(new Uri(activity.ServiceUrl));
                // calculate something for us to return
                string Answer;
                StockLUIS StLUIS= await GetEntityFromLUIS(activity.Text);

                if (StLUIS.intents.Count() > 0)
                {
                    switch (StLUIS.intents[0].intent)
                    {
                        case "getstock":
                            Answer = await GetStock(StLUIS.entities[0].entity);
                            break;
                        case "getgraph":
                            Answer = @"feature will be added soon";
                            break;
                        case "getprediction":
                            Answer = await GetPrediction(StLUIS.entities);
                            break;
                        case "highlow":
                            Answer = @"cant say right now..feature will be added soon";
                            break;
                        default:
                            Answer = @"Sorry, I am not getting you...";
                            break;
                    }
                }
                else
                {
                    Answer = "Sorry, I am not getting you...";
                }

                

                // return our reply to the user
                Activity reply = activity.CreateReply(Answer);  
                await connector.Conversations.ReplyToActivityAsync(reply);
            }
            else
            {
                ConnectorClient connector = new ConnectorClient(new Uri(activity.ServiceUrl));
                Activity reply=HandleSystemMessage(activity);
                await connector.Conversations.ReplyToActivityAsync(reply);
            }
            var response = Request.CreateResponse(HttpStatusCode.OK);
            return response;
        }
        public async Task<string> GetPrediction(Entity[] arr)
        {
            string comp=" ", date= " ";
            
            for (int i = 0;  i < arr.Length; ++i)
            {
                if (arr.Length!=2)
                {
                    return @"please specifiy both date and CompanyCode for opening stock value prediction";
                }                           
                if(arr[i].type==@"company tag")
                {
                    comp = arr[i].entity;
                }
                else if (arr[i].type==@"datetime")
                {
                    date = arr[i].entity;
                }
                
            }
            string sqlquery = @"SELECT predicted_open_price FROM " + comp + @" WHERE date='" + date + "'";
            string x= await AccessDatabase(sqlquery);
            string x1 = @"predicted opening stock price for " + comp + @" is " + x;
            return x1;
        }
        private Activity  HandleSystemMessage(Activity message)
        {
            if (message.Type == ActivityTypes.DeleteUserData)
            {
                // Implement user deletion here
                // If we handle user deletion, return a real message
                Activity reply = message.CreateReply(string.Format("{0} deleted", message.From.Id));  // might add a $
                return reply;

            }
            else if (message.Type == ActivityTypes.ConversationUpdate)
            {
                // Handle conversation state changes, like members being added and removed
                // Use Activity.MembersAdded and Activity.MembersRemoved and Activity.Action for info
                // Not available in all channels

                Activity reply = message.CreateReply(string.Format("{0} {1}", message.MembersAdded.ToString(),message.Action));  // might add a $
                return reply;
            }
            else if (message.Type == ActivityTypes.ContactRelationUpdate)
            {
                // Handle add/remove from contact lists
                // Activity.From + Activity.Action represent what happened
            }
            else if (message.Type == ActivityTypes.Typing)
            {
                // Handle knowing tha the user is typing

            }
            else if (message.Type == ActivityTypes.Ping)
            {
                
            }

            return null;
        }
        private static async Task<StockLUIS> GetEntityFromLUIS(string Query)
        {
            Query = Uri.EscapeDataString(Query);
            StockLUIS Data = new StockLUIS();
            using (HttpClient client = new HttpClient())
            {
                string RequestURI = "https://api.projectoxford.ai/luis/v1/application?id=3afa3cd9-a634-40b9-a3fa-1c0d09d9195f&subscription-key=afdf376548c1426b830453f5c0a66c7e&q=" + Query;
                HttpResponseMessage msg = await client.GetAsync(RequestURI);

                if (msg.IsSuccessStatusCode)
                {
                    var JsonDataResponse = await msg.Content.ReadAsStringAsync();
                    Data = JsonConvert.DeserializeObject<StockLUIS>(JsonDataResponse);
                }
            }
            return Data;
        }
        private async Task<string> GetStock(string StockSymbol)
        {
            double? dblStockValue = await YahooBot.GetStockRateAsync(StockSymbol);
            if (dblStockValue == null)
            {
                return string.Format("This \"{0}\" is not an valid stock symbol", StockSymbol);
            }
            else
            {
                return string.Format("Stock Price of {0} is {1}", StockSymbol, dblStockValue);
            }
        }
        static public async Task<string> AccessDatabase(string command)
        {
            //throw new TestSqlException(4060); //(7654321);  // Uncomment for testing.  

            using (var sqlConnection = new QC.SqlConnection
                (GetSqlConnectionString()))
            {
                using (var dbCommand = sqlConnection.CreateCommand())
                {
                    dbCommand.CommandText = command;

                    sqlConnection.Open();
                    var dataReader = dbCommand.ExecuteReader();
                    sqlConnection.Close();
                    await Task.Delay(10000);
                    return dataReader.GetString(0);
                    
                }
            }
        }
        static private string GetSqlConnectionString()
        {
            // Prepare the connection string to Azure SQL Database.  
            var sqlConnectionSB = new QC.SqlConnectionStringBuilder();

            // Change these values to your values.  
            sqlConnectionSB.DataSource = "tcp:rathoreserver.database.windows.net,1433"; //["Server"]  
            sqlConnectionSB.InitialCatalog = "stock2data"; //["Database"]  

            sqlConnectionSB.UserID = "aditya";  // "@yourservername"  as suffix sometimes.  
            sqlConnectionSB.Password = "HTMLc++java";
            sqlConnectionSB.IntegratedSecurity = false;

            // Adjust these values if you like. (ADO.NET 4.5.1 or later.)  
            sqlConnectionSB.ConnectRetryCount = 3;
            sqlConnectionSB.ConnectRetryInterval = 10;  // Seconds.  

            // Leave these values as they are.  
            sqlConnectionSB.IntegratedSecurity = false;
            sqlConnectionSB.Encrypt = true;
            sqlConnectionSB.ConnectTimeout = 30;

            return sqlConnectionSB.ToString();
        }

    }
}